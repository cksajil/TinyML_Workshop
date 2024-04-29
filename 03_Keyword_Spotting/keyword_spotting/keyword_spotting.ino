// Requires Arduino CMSIS-DSP Ver.5.7.0 Library
#include <PDM.h>
#include "model.h"
#include <arm_math.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#define SAMPLES 1024
#define SAMPLING_FREQUENCY 16000

volatile int yesCounter;
volatile int noCounter;
short sampleBuffer[SAMPLES];
float32_t sampleBufferFloat[SAMPLES];
volatile int samplesRead;

void onPDMdata(void);

static const char CHANNELS = 1;
const int WINDOW_SIZE = 1024;

q15_t input_q15[WINDOW_SIZE];
q15_t hanning_window_q15[WINDOW_SIZE];
q15_t processed_window_q15[WINDOW_SIZE];

const int ROWS = 7;
const int COLS = 129;

arm_rfft_instance_q15 S_q15;

namespace tflite
{
    namespace ops
    {
        namespace micro
        {
            TfLiteRegistration *Register_CONV_2D();
            // TfLiteRegistration *Register_DEPTHWISE_CONV_2D();
            TfLiteRegistration *Register_FULLY_CONNECTED();
            // TfLiteRegistration *Register_FULLY_SOFTMAX();
            TfLiteRegistration *Register_SOFTMAX();
        }
    }
}

namespace
{
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *model_input = nullptr;
    TfLiteTensor *model_output = nullptr;

    // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 2048;
    uint8_t tensor_arena[kTensorArenaSize];
}

// this is twice the size because each FFT output has a real and imaginary part
q15_t fft_q15[WINDOW_SIZE * 2];

// this is half the size of WINDOW_SIZE becase we just need the magnitude from
// the first half of the FFT output
q15_t fft_mag_q15[WINDOW_SIZE / 2];

void setup()
{
    Serial.begin(9600);
    while (!Serial)
    {
    };

    PDM.onReceive(onPDMdata);
    PDM.setBufferSize(SAMPLES);
    PDM.setGain(40);
    if (!PDM.begin(CHANNELS, SAMPLING_FREQUENCY))
    {
        Serial.println("Failed to start PDM!");
    }
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("Schema mismatch");

        return;
    }
    // Pull in only the operation implementations we need.
    tflite::AllOpsResolver tflOpsResolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, tflOpsResolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        Serial.println("AllocateTensors() failed");
        return;
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    // pipeline initialization
    hanning_window_init_q15(hanning_window_q15, WINDOW_SIZE);
    arm_rfft_init_q15(&S_q15, WINDOW_SIZE, 0, 1);
    yesCounter = 0;
    noCounter = 0;
}

void loop()
{   
    if (samplesRead)
    {
        for (int i = 0; i < samplesRead; i++)
        {
            sampleBufferFloat[i] = static_cast<float32_t>(sampleBuffer[i]);
        }
        // for (int i = 0; i < samplesRead; i++)
        // {
        //     Serial.println(sampleBufferFloat[i]);
        // }

        // initialize input with Sample Buffer Values
        for (int i = 0; i < samplesRead; i++)
        {
            arm_float_to_q15(&sampleBufferFloat[i], &input_q15[i], 1);
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

        //  Magnitude of FFT computed is available in fft_mag_q15
        // for (int i = 0; i < WINDOW_SIZE; i++)
        // {
        //     Serial.println(fft_mag_q15[i]);
        // }

        // clear the read count
        samplesRead = 0;
    }

    // for (int i = 0; i < ROWS; i++)
    // {
    //     for (int j = 0; j < COLS; j++)
    //     {
    //         model_input->data.f[i * COLS + j] = static_cast<float32_t>(fft_mag_q15[j]);
    //     }
    // }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        model_input->data.f[i] = static_cast<float32_t>(fft_mag_q15[i]);
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        Serial.println("Invoke failed");
        return;
    }

    float pred_0 = model_output->data.f[0];
    float pred_1 = model_output->data.f[1];

    if (pred_0 == 1.0)
    {
      yesCounter = yesCounter+1;
    }
    else
    {
      noCounter = noCounter+1;
    }

    if (yesCounter>200)
    {
        Serial.println("Class predicted is Yes ");
        yesCounter = 0;
    }

    if (noCounter>200)
    {
        Serial.println("Class predicted is No ");
        noCounter = 0;
    }

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
