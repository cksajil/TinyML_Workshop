#include <TensorFlowLite.h>
#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace
{

    int32_t previous_time = 0;
    FeatureProvider *feature_provider = nullptr;
    constexpr int kTensorArenaSize = 10 * 1024;
    // Keep aligned to 16 bytes for CMSIS
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    int8_t feature_buffer[kFeatureElementCount];

} // namespace

void setup()
{
    delay(1000);
    previous_time = 0;

    // start the audio
    TfLiteStatus init_status = InitAudioRecording();
    if (init_status != kTfLiteOk)
    {
        MicroPrintf("Unable to initialize audio");
        return;
    }

    MicroPrintf("Initialization complete");

    static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                   feature_buffer);
    feature_provider = &static_feature_provider;
}

void loop()
{
    const int32_t current_time = LatestAudioTimestamp();
    int how_many_new_slices = 0;

    // MicroPrintf("Sajil is trying to print features computed");

    TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
        previous_time, current_time, &how_many_new_slices);
    if (feature_status != kTfLiteOk)
    {
        MicroPrintf("Feature generation failed");
        return;
    }
    previous_time += how_many_new_slices * kFeatureSliceStrideMs;
    // If no new audio samples have been received since last time, don't bother
    // running the network model.
    if (how_many_new_slices == 0)
    {
        return;
    }

    // for (int i = 0; i < kFeatureElementCount; i++)
    // {
    //     MicroPrintf("%d", feature_buffer[i]);
    // }
}