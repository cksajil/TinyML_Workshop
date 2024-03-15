#include <Arduino.h>
#include "model.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Define your test input array
float test_input[61][129];

void setup()
{
    // Initialize serial communication
    Serial.begin(9600);

    // Initialize TensorFlow Lite interpreter
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interpreter(model_data, resolver, micro_error_reporter);
    interpreter.AllocateTensors();

    // Perform inference on the test input array
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    // Load test input into input tensor
    for (int i = 0; i < 61; i++)
    {
        for (int j = 0; j < 129; j++)
        {
            input->data.f[i * 129 + j] = test_input[i][j];
        }
    }

    // Invoke interpreter
    interpreter.Invoke();

    // Print inference results
    for (int i = 0; i < output->dims->data[0]; i++)
    {
        Serial.print(output->data.f[i]);
        Serial.print(" ");
    }
}

void loop()
{
    // Nothing to do here as we only perform inference once in setup
}
