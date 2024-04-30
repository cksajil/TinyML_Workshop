#include <PDM.h>
#include "mfcc.h"

#define SAMPLES 320
#define SAMPLING_FREQUENCY 16000
#define NUM_FRAMES 150
#define NUM_MFCC_COEFFS 10
#define MFCC_DEC_BITS 4
#define FRAME_SHIFT_MS 20
#define FRAME_SHIFT ((int16_t)(SAMPLING_FREQUENCY * 0.001 * FRAME_SHIFT_MS))
#define MFCC_BUFFER_SIZE (NUM_FRAMES * NUM_MFCC_COEFFS)
#define FRAME_LEN_MS 20
#define FRAME_LEN ((int16_t)(SAMPLING_FREQUENCY * 0.001 * FRAME_LEN_MS))

int recording_win = NUM_FRAMES;
short audio_buffer[SAMPLES];
int16_t converted_buffer[SAMPLES];
volatile int samplesRead;

int num_frames = NUM_FRAMES;
int num_mfcc_features = NUM_MFCC_COEFFS;
int frame_shift = FRAME_SHIFT;
int frame_len = FRAME_LEN;
int mfcc_dec_bits = MFCC_DEC_BITS;

MFCC *mfcc;
float *mfcc_buffer;

void onPDMdata(void);
void extract_features(int16_t *audio_buffer, float *mfcc_buffer);

void setup()
{
    Serial.begin(9600);
    while (!Serial)
    {
    }

    PDM.onReceive(onPDMdata);
    PDM.setBufferSize(NUM_FRAMES);
    PDM.setGain(40);
    if (!PDM.begin(1, SAMPLING_FREQUENCY))
    {
        Serial.println("Failed to start PDM!");
    }

    mfcc = new MFCC(num_mfcc_features, frame_len, mfcc_dec_bits);
    mfcc_buffer = new float[NUM_FRAMES * NUM_MFCC_COEFFS];
}

void loop()
{
    if (samplesRead)
    {
        for (size_t i = 0; i < SAMPLES; i++)
        {
            converted_buffer[i] = (int16_t)audio_buffer[i];
        }
        extract_features(converted_buffer, mfcc_buffer);

        for (int i = 0; i < MFCC_BUFFER_SIZE; i++)
        {
            Serial.println(mfcc_buffer[i]);
        }

        samplesRead = 0;
    }
}

void extract_features(int16_t *audio_buffer, float *mfcc_buffer)
{
    int32_t mfcc_buffer_head = (num_frames - recording_win) * num_mfcc_features;
    mfcc->mfcc_compute(audio_buffer, &mfcc_buffer[mfcc_buffer_head]);
}

void onPDMdata()
{
    int bytesRead = PDM.read((uint8_t *)audio_buffer, NUM_FRAMES * sizeof(short));
    samplesRead = bytesRead / sizeof(short);
}
