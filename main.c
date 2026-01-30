#include <stdlib.h>
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "kiss_fftr.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#define LN_OF_A (0.057762265046662109118102676788181380672958344530021271176723334124449468497474559633821943916368224)
#define F0 (440)

struct userdata {
  float *samples;
  const float *window;
  float *workbuffer;
  kiss_fft_cpx *fft_out;
  kiss_fftr_cfg *fft_config;
  const size_t fft_rate;
  size_t write_pos;
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

  ma_uint32 halfCount = userdata->fft_rate/2 + 1;
  for (int i = 0; i < halfCount; i++) {
    kiss_fft_cpx c = userdata->fft_out[i];
    float magnitude = fmaxf(sqrtf(c.r * c.r + c.i * c.i), 1e-12);
    userdata->workbuffer[i] = magnitude;
    userdata->workbuffer[i+halfCount] = logf(magnitude);
  }

  for (int h = 2; h <= 4; h++)
    for (int i = 0; i * h < halfCount; i++)
      userdata->workbuffer[i+halfCount] += logf(userdata->workbuffer[i*h]);

  static const float fmin = 130.0f;
  static const float fmax = 1000.0f;

  const int k_min = (int)(fmin * halfCount / pDevice->sampleRate);
  const int k_max = (int)(fmax * halfCount / pDevice->sampleRate);

  int idx;
  float max = -INFINITY;

  for (int i = halfCount + k_min; i < halfCount + k_max; i++) {
  // for (int i = k_min; i < k_max; i++) {
    if (userdata->workbuffer[i] > max) {
      idx = i;
      max = userdata->workbuffer[i];
    }
  }

  float delta = 0.0f;
  if (idx > halfCount && idx < halfCount*2-1) {
    float alpha = userdata->workbuffer[idx-1];
    float beta  = userdata->workbuffer[idx];
    float gamma = userdata->workbuffer[idx+1];

    delta = 0.5f * (alpha - gamma) / (alpha - 2.0f * beta + gamma);
  }

  float i_precise = idx - halfCount + delta;

  float pitch_hz = i_precise * pDevice->sampleRate / userdata->fft_rate;

  float rms = 0.0f;
  for (int i = 0; i < userdata->fft_rate; i++)
    rms += userdata->samples[i] * userdata->samples[i];

  rms = sqrtf(rms / userdata->fft_rate);

  if (rms < 0.001f) {
    pitch_hz = 0.0f;   // or “no pitch”
    return;
  }

  int n = half_steps(pitch_hz);
  
  const static char *notes[12] = {"A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"};
  
  printf("\rFC:%5d | %2s%d (%5.1f)", frameCount, notes[((n%12) + 12) % 12], (n+57)/12, pitch_hz);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  /*
    TODO:
    - parse args for sample rate, fft rate and device
    - do device enumeration for picking device
  */

  ma_result result;

  int fft_rate = 8192*2;
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
    0
  };
  
  ma_device_config device_config = ma_device_config_init(ma_device_type_capture);
  device_config.capture.calculateLFEFromSpatialChannels = MA_TRUE;
  device_config.capture.channelMixMode = ma_channel_mix_mode_rectangular;
  device_config.capture.channels = 1;
  device_config.capture.format = ma_format_f32;
  device_config.dataCallback = data_callback;
  device_config.sampleRate = 0;
  device_config.pUserData = &userdata;
  
  ma_device device;
  result = ma_device_init(NULL, &device_config, &device);
  if (result != MA_SUCCESS) {
    printf("Error %d opening device", result);
    return 1;
  }

  printf("Device sample rate: %d\n", device.sampleRate);

  ma_device_start(&device);

  getchar();

  ma_device_uninit(&device);
  free(samples);
  free(fft_out);
  free(fftr_cfg);
  
  return 0;
}
