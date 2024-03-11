// Requires Arduino CMSIS-DSP Ver.5.7.0 Library
#include <arm_math.h>
#include <PDM.h>
#include "model.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// / Globals, used for compatibility with Arduino-style sketches.
namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *model_input = nullptr;
    TfLiteTensor *model_output = nullptr;
    // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 10 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
    int8_t *model_input_buffer = nullptr;
}

// Constants & Globals
#define SAMPLES 256
#define SAMPLING_FREQUENCY 16000
#define NROWS 61
#define NCOLS 129
float32_t sampleBuffer[SAMPLES];

volatile int samplesRead;
void onPDMdata(void);

static const char CHANNELS = 1;

const int WINDOW_SIZE = 256;
const int STEP_SIZE = 128;

q15_t input_q15[WINDOW_SIZE];
q15_t hanning_window_q15[WINDOW_SIZE];
q15_t processed_window_q15[WINDOW_SIZE];

arm_rfft_instance_q15 S_q15;

// This is twice the size because each FFT output has a real and imaginary part
q15_t fft_q15[WINDOW_SIZE * 2];

// this is half the size of WINDOW_SIZE becase we just need the magnitude from
// the first half of the FFT output
q15_t fft_mag_q15[WINDOW_SIZE / 2];

void setup()
{
    delay(3000);
    Serial.begin(9600);
    while (!Serial)
        ;

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    tflite::ops::micro::AllOpsResolver resolver;
    PDM.onReceive(onPDMdata);
    PDM.setGain(40);

    if (!PDM.begin(CHANNELS, SAMPLING_FREQUENCY))
    {
        Serial.println("Failed to start PDM!");
    }

    // pipeline initialization
    hanning_window_init_q15(hanning_window_q15, WINDOW_SIZE);
    arm_rfft_init_q15(&S_q15, WINDOW_SIZE, 0, 1);

    const tflite::Model *model = ::tflite::GetModel(model);
    // if (model->version() != TFLITE_SCHEMA_VERSION)
    // {
    //     TF_LITE_REPORT_ERROR(error_reporter,
    //                          "Model provided is schema version %d not equal "
    //                          "to supported version %d.",
    //                          model->version(), TFLITE_SCHEMA_VERSION);
    //     return;
    // }

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }
    // Get information about the memory area to use for the model's input.
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);
}

void loop()
{
    // initialize input with Sample Buffer Values
    for (int i = 0; i < samplesRead; i++)
    {
        arm_float_to_q15(&sampleBuffer[i], &input_q15[i], 1);
    }
    // unsigned long pipeline_start_us = micros();

    // equivalent to: processed_window_q15 = input_q15 * hanning_window_q15
    arm_mult_q15(input_q15, hanning_window_q15, processed_window_q15, WINDOW_SIZE);

    // calculate the FFT and FFT magnitude
    arm_rfft_q15(&S_q15, processed_window_q15, fft_q15);
    arm_cmplx_mag_q15(fft_q15, fft_mag_q15, WINDOW_SIZE / 2);

    // unsigned long pipeline_end_us = micros();

    // Serial.print("Pipeline run time = ");
    // Serial.print(pipeline_end_us - pipeline_start_us);
    // Serial.println(" microseconds");

    //  Maggitude of FFT computed is available in fft_mag_q15

    // input tensor

    // Load test input into input tensor
    for (int j = 0; j < NROWS; j++)
    {
        for (int k = 0; k < NCOLS; k++)
        {
            input->data.f[j * 129 + k] = fft_mag_q15[j][k];
        }
    }

    TfLiteStatus invokeStatus = interpreter->Invoke();
    if (invokeStatus != kTfLiteOk)
    {
        Serial.println("Invoke failed!");
        while (1)
            ;
        return;
    }
    Serial.println(model_output->data.f[0]);

    samplesRead = 0;
}

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

void onPDMdata()
{
    int bytesAvailable = PDM.available();
    PDM.read(sampleBuffer, bytesAvailable);
    samplesRead = bytesAvailable / 2;
}