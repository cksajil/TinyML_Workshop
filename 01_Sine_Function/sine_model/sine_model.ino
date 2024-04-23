#include "model.h"
#include <arm_math.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace tflite
{
    namespace ops
    {
        namespace micro
        {

            TfLiteRegistration *Register_FULLY_CONNECTED();
        }
    }
}

namespace
{
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *model_input = nullptr;
    TfLiteTensor *model_output = nullptr;

    constexpr int kTensorArenaSize = 2 * 2048;
    uint8_t tensor_arena[kTensorArenaSize];

    float sinFrequency = 1; // Default frequency
    int numSamples = 1000;  // Default number of samples
}

void setup()
{
    Serial.begin(9600);
    while (!Serial)
    {
    };

    // initialize digital pin LED_BUILTIN as an output.
    pinMode(LED_BUILTIN, OUTPUT);

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
}

void loop()
{

    for (int i = 0; i < numSamples; i++)
    {

        float x = 2 * PI * sinFrequency * i / numSamples;

        // Library Implementation of Sine Function
        // float y_pred = sin(x);

        // ML Approximation of Sine Function
        model_input->data.f[0] = x;

        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk)
        {
            Serial.println("Invoke failed");
            return;
        }

        float y_pred = model_output->data.f[0];

        Serial.println(y_pred);

        int ledState = y_pred * 255;
        analogWrite(LED_BUILTIN, ledState);
    }
}
