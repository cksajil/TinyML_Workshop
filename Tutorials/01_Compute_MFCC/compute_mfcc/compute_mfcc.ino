#include <PDM.h>

#define SAMPLING_FREQUENCY 16000
#define NUM_SAMPLES 512
short sampleBuffer[NUM_SAMPLES];
volatile int samplesRead;

void onPDMdata(void);

void setup()
{
    Serial.begin(9600);
    while (!Serial)
    {
    };

    PDM.onReceive(onPDMdata);
    PDM.setBufferSize(NUM_SAMPLES);
    PDM.setGain(40);
    if (!PDM.begin(1, SAMPLING_FREQUENCY))
    {
        Serial.println("Failed to start PDM!");
    }
}

void loop()
{
    if (samplesRead)
    {
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            Serial.println(sampleBuffer[i]);
        }
        samplesRead = 0;
    }
}

void onPDMdata()
{
    int bytesRead = PDM.read((uint8_t *)sampleBuffer, NUM_SAMPLES * sizeof(short));
    samplesRead = bytesRead / sizeof(short);
}
