/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <arm_math.h>
#include "feature_provider.h"

#include "audio_provider.h"
#include "micro_features_micro_features_generator.h"
#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"

const int WINDOW_SIZE = 256;
float32_t sampleBuffer[WINDOW_SIZE];

q15_t input_q15[WINDOW_SIZE];
q15_t hanning_window_q15[WINDOW_SIZE];
q15_t processed_window_q15[WINDOW_SIZE];

arm_rfft_instance_q15 S_q15;
q15_t fft_q15[WINDOW_SIZE * 2];
q15_t fft_mag_q15[WINDOW_SIZE / 2];

void hanning_window_init_q15(q15_t *hanning_window_q15, size_t size)
{
  for (size_t i = 0; i < size; i++)
  {
    // calculate the Hanning Window value for i as a float32_t
    float32_t f = 0.5 * (1.0 - arm_cos_f32(2 * PI * i / size));

    // convert value for index i from float32_t to q15_t and store
    // in window at position i
    arm_float_to_q15(&f, &hanning_window_q15[i], 1);
  }
}

FeatureProvider::FeatureProvider(int feature_size, int8_t *feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true)
{
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n)
  {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(int32_t last_time_in_ms,
                                                  int32_t time_in_ms,
                                                  int *how_many_new_slices)
{
  // MicroPrintf("Inside Feature Provider");

  hanning_window_init_q15(hanning_window_q15, WINDOW_SIZE);
  arm_rfft_init_q15(&S_q15, WINDOW_SIZE, 0, 1);

  if (feature_size_ != kFeatureElementCount)
  {
    MicroPrintf("Requested feature_data_ size %d doesn't match %d",
                feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
  // Number of new 20ms slices from which we can take 30ms samples
  int slices_needed =
      ((((time_in_ms - last_time_in_ms) - kFeatureSliceDurationMs) *
        kFeatureSliceStrideMs) /
           kFeatureSliceStrideMs +
       kFeatureSliceStrideMs) /
      kFeatureSliceStrideMs;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_)
  {
    TfLiteStatus init_status = InitializeMicroFeatures();
    if (init_status != kTfLiteOk)
    {
      return init_status;
    }
    is_first_run_ = false;
    return kTfLiteOk;
  }
  if (slices_needed > kFeatureSliceCount)
  {
    slices_needed = kFeatureSliceCount;
  }
  if (slices_needed == 0)
  {
    return kTfLiteOk;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0)
  {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice)
    {
      int8_t *dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t *src_slice_data =
          feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i)
      {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0)
  {
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice)
    {
      const int new_step = last_step + (new_slice - slices_to_keep);
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
      int16_t *audio_samples = nullptr;
      int audio_samples_size = 0;
      GetAudioSamples(slice_start_ms, kFeatureSliceDurationMs,
                      &audio_samples_size, &audio_samples);
      constexpr int wanted =
          kFeatureSliceDurationMs * (kAudioSampleFrequency / 1000);
      if (audio_samples_size != wanted)
      {
        MicroPrintf("Audio data size %d too small, want %d", audio_samples_size,
                    wanted);
        return kTfLiteError;
      }
      int8_t *new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
      size_t num_samples_read;

      // MicroPrintf("Got Audio Samples");

      // // Code to print raw audio samples
      // for (int i = 0; i < audio_samples_size; i++)
      // {
      //   MicroPrintf("%d", audio_samples[i]);
      // }

      for (int i = 0; i < audio_samples_size; i++)
      {
        sampleBuffer[i] = audio_samples[i];
      }

      for (int i = 0; i < audio_samples_size; i++)
      {
        arm_float_to_q15(&sampleBuffer[i], &input_q15[i], 1);
      }

      arm_mult_q15(input_q15, hanning_window_q15, processed_window_q15, WINDOW_SIZE);
      // calculate the FFT and FFT magnitude
      arm_rfft_q15(&S_q15, processed_window_q15, fft_q15);
      arm_cmplx_mag_q15(fft_q15, fft_mag_q15, WINDOW_SIZE / 2);

      //  Maggitude of FFT computed is available in fftComputed
      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        MicroPrintf("%d", fft_mag_q15[i]);
      }

      // for (int i = 0; i < audio_samples_size; i++)
      // {
      //   MicroPrintf("%f", sampleBuffer[i]);
      // }

      // MicroPrintf("Conversion success for audio samples");

      // // Code to print size of audio samples (sampling_rate*30ms)
      // MicroPrintf("%d", audio_samples_size);

      // TfLiteStatus generate_status = GenerateMicroFeatures(
      //     audio_samples, audio_samples_size, kFeatureSliceSize, new_slice_data,
      //     &num_samples_read);
      // if (generate_status != kTfLiteOk)
      // {
      //   return generate_status;
      // }
    }
  }
  return kTfLiteOk;
}
