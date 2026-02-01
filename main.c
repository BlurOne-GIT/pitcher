#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "kiss_fftr.h"

#define FLAG_IMPLEMENTATION
#include "flag.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define LN_OF_A (0.057762265046662109118102676788181380672958344530021271176723334124449468497474559633821943916368224)
#define F0 (440)

struct userdata {
  float *samples;
  const float *window;
  float *workbuffer;
  kiss_fft_cpx *fft_out;
  kiss_fftr_cfg *fft_config;
  const ma_uint32 fft_rate;
  const ma_uint32 half_count;
  const float rms_min;
  size_t write_pos;
  size_t k_min;
  size_t k_max;
};

static int half_steps(float fn) {
  return (int)roundf(log(fn/F0)/LN_OF_A);
}

void data_callback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 frameCount) {
  struct userdata *userdata = (struct userdata *)pDevice->pUserData;

  if (userdata->write_pos + frameCount >= userdata->fft_rate) {
    size_t keep = userdata->fft_rate - frameCount;
    memmove(
      userdata->samples,
      userdata->samples + (userdata->write_pos - keep),
      keep*sizeof(float)
    );
    userdata->write_pos = keep;
  }
  
  memcpy(userdata->samples + userdata->write_pos, pInput, frameCount*sizeof(float));
  userdata->write_pos += frameCount;

  if (userdata->write_pos < userdata->fft_rate) return;
  
  for (int i = 0; i < userdata->fft_rate; i++)
    userdata->workbuffer[i] = userdata->samples[i] * userdata->window[i];

  kiss_fftr(*userdata->fft_config, userdata->workbuffer, userdata->fft_out);

  for (int i = 0; i < userdata->half_count; i++) {
    kiss_fft_cpx c = userdata->fft_out[i];
    float magnitude = fmaxf(sqrtf(c.r * c.r + c.i * c.i), 1e-12);
    userdata->workbuffer[i+userdata->half_count] = magnitude;
    userdata->workbuffer[i] = logf(magnitude);
  }

  for (int h = 2; h <= 4; h++)
    for (int i = 0; i * h < userdata->half_count; i++)
      userdata->workbuffer[i] += logf(userdata->workbuffer[userdata->half_count+i*h]);

  int idx;
  float max = -INFINITY;

  for (int i = userdata->k_min; i < userdata->k_max; i++) {
    if (userdata->workbuffer[i] > max) {
      idx = i;
      max = userdata->workbuffer[i];
    }
  }

  if (max < 0) return;

  float delta = 0.0f;
  if (idx > 0 && idx < userdata->half_count-1) {
    float alpha = userdata->workbuffer[idx-1];
    float beta  = userdata->workbuffer[idx];
    float gamma = userdata->workbuffer[idx+1];

    delta = 0.5f * (alpha - gamma) / (alpha - 2.0f * beta + gamma);
  }

  float i_precise = idx + delta;

  float pitch_hz = i_precise * pDevice->sampleRate / userdata->fft_rate;

  float rms = 0.0f;
  for (int i = 0; i < userdata->fft_rate; i++)
    rms += userdata->samples[i] * userdata->samples[i];

  rms = sqrtf(rms / userdata->fft_rate);

  static int n;
  if (rms >= userdata->rms_min) {
    n = half_steps(pitch_hz);
  }

  if (n + 57 < 0) return;
    
  const static char *notes[12] = {
    "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"
  };
  
  printf("\r%2s%1d (%6.1fHz) | Energy: %f", notes[((n%12) + 12) % 12], (n+57)/12, pitch_hz, rms);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  uint64_t *device_id = flag_uint64("-device", 0, "Use a specific device for the detection.");
  uint64_t *sample_rate = flag_uint64("-sampleRate", 0, "The sample rate to use for the device. It uses the device's prefered sample rate by default.");
  uint64_t *fft_rate_pow = flag_uint64("-fftRate", 13, "2^value of samples to use for the fft.");
  float *fmin = flag_float("-highpass", 80, "Lowest frequency (in Hz) for the high-pass filter.");
  float *fmax = flag_float("-lowpass", 1110, "Highest frequency (in Hz) for the low-pass filter.");
  float *rms_min = flag_float("-rms", .001f, "Minimum rms for loudness filter.");
  _Bool *help = flag_bool("-help", 0, "Display this help message.");

  if (!flag_parse(argc, argv) || *help) {
    fprintf(stderr, "Usage: %s [OPTIONS]\nOPTIONS:\n", argv[0]);
    flag_print_options(stderr);
    if (!*help)
      flag_print_error(stderr);
    return !*help;
  }

  int fft_rate = pow(2, *fft_rate_pow);
  float *samples = malloc(fft_rate * sizeof(float));
  float *window = malloc(fft_rate*sizeof(float));
  float *workbuffer = malloc((fft_rate + 2) * sizeof(float));
  kiss_fft_cpx *fft_out = malloc((fft_rate/2+1)*sizeof(kiss_fft_cpx));
  kiss_fftr_cfg fftr_cfg = kiss_fftr_alloc(fft_rate, 0, NULL, NULL);
 
  for (int i = 0; i < fft_rate; i++)
    window[i] = 0.5f * (1 - cosf(2*M_PI*i/(fft_rate-1)));
  
  struct userdata userdata = {
    samples,
    window,
    workbuffer,
    fft_out,
    &fftr_cfg,
    fft_rate,
    fft_rate/2 + 1,
    *rms_min,
    0
  };

  ma_result result;
  
  ma_device_config device_config = ma_device_config_init(ma_device_type_capture);
  device_config.capture.calculateLFEFromSpatialChannels = MA_TRUE;
  device_config.capture.channelMixMode = ma_channel_mix_mode_rectangular;
  device_config.capture.channels = 1;
  device_config.capture.format = ma_format_f32;
  device_config.dataCallback = data_callback;
  device_config.sampleRate = *sample_rate;
  device_config.pUserData = &userdata;
  
  ma_context context;
  result = ma_context_init(NULL, 0, NULL, &context);
  if (result != MA_SUCCESS) {
    fprintf(stderr, "Error %d creating context\n", result);
    return 2;
  }
  
  if (*device_id != 0) {
    ma_device_info* pCaptureInfos;
    ma_uint32 captureCount;
    result = ma_context_get_devices(&context, NULL, NULL, &pCaptureInfos, &captureCount);
    if (result != MA_SUCCESS) {
      fprintf(stderr, "Error %d getting devices\n", result);
      return 3;
    }

    if (*device_id >= captureCount) {
      fprintf(stderr, "Device #%lu not found.\n", *device_id);
      printf("Avaiable devices:\n");
      for (ma_uint32 iDevice = 0; iDevice < captureCount; iDevice++) {
        if (pCaptureInfos[iDevice].isDefault)
          printf("(Default)");
        else
          printf("         ");
        printf(" %d. %s\n", iDevice+1, pCaptureInfos[iDevice].name);
      }
      return 4;
    }

    device_config.capture.pDeviceID = &pCaptureInfos[*device_id-1].id;
  }
  
  ma_device device;
  result = ma_device_init(&context, &device_config, &device);
  if (result != MA_SUCCESS) {
    fprintf(stderr, "Error %d opening device\n", result);
    return 1;
  }

  if (!*sample_rate)
    printf("Device sample rate: %d\n", device.sampleRate);

  userdata.k_min = round(*fmin * userdata.half_count / device.sampleRate);  
  userdata.k_max = round(*fmax * userdata.half_count / device.sampleRate);  

  ma_device_start(&device);

  getchar();

  ma_device_uninit(&device);
  
  return 0;
}
