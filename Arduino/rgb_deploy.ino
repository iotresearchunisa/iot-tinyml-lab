#include "aifes.h"

#define s0 8
#define s1 9
#define s2 10
#define s3 11
#define out 12

int array_max(const float *arr, int len) {
    int idx = 0;
    float max_val = arr[idx];

    for (int i = 1; i < len; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            idx = i;
        }
    }
    return idx;
}

float weights_data_dense_1[48] = {-0.560545f,0.511459f,-0.344696f,-0.054459f,-0.912129f,-0.765001f,-0.644087f,1.215726f,0.035258f,0.057757f,-0.498377f,-0.153616f,-0.410332f,-0.138117f,-0.220291f,-1.192695f,-0.459223f,-0.649363f,-0.396386f,-0.694785f,-0.398034f,-0.748121f,-0.678582f,-0.721324f,-1.088555f,0.674034f,-0.423683f,-0.497269f,-0.548034f,-0.386663f,-0.375120f,-0.485800f,-0.055308f,0.649001f,-0.497845f,0.117303f,-0.241125f,-0.380345f,-0.495794f,-0.462417f,-0.151413f,-0.458786f,0.853500f,-0.361590f,0.020927f,-1.323016f,-0.042618f,-0.493374f};
float bias_data_dense_1[16] = {0.000000f,1.758869f,0.000000f,-0.250408f,-1.052887f,-0.730911f,-0.164676f,1.561766f,-0.499435f,-0.818920f,0.334929f,-0.700227f,-0.575940f,-0.837277f,0.000000f,-0.659430f};
float weights_data_dense_2[128] = {0.362239f,-0.290399f,0.279656f,0.343654f,0.496796f,0.499695f,0.111499f,-0.107562f,-1.038948f,-0.739391f,0.434657f,-1.181108f,0.086928f,-0.725299f,-0.090765f,-0.938896f,-0.491211f,0.418790f,-0.224113f,-0.227104f,0.087909f,0.191183f,0.337611f,0.226493f,-0.201421f,-0.895191f,0.051582f,0.085893f,0.058261f,0.000099f,0.590214f,-0.890930f,0.075562f,-0.715315f,-0.325247f,-0.076382f,-0.382278f,0.269329f,-0.781016f,-0.274895f,-0.677643f,-0.297669f,0.593927f,0.210012f,-0.342404f,0.408617f,-0.198391f,0.157308f,-0.008103f,-0.436460f,0.708058f,0.004807f,-0.352504f,-0.043019f,-0.358425f,0.405118f,-0.069178f,-0.254864f,-0.780101f,-0.571514f,0.618395f,-0.669399f,-0.196459f,-0.137537f,-0.377915f,-0.518501f,0.074341f,-0.691656f,-0.059635f,-0.184400f,0.130724f,0.003922f,-0.080368f,-0.741519f,-0.406510f,-0.145829f,-0.136659f,0.526347f,-0.245535f,-0.537247f,-1.173749f,-0.732890f,0.746575f,-0.805223f,-0.829274f,0.011136f,-0.203686f,-0.395322f,0.101249f,-0.179165f,-0.371961f,-0.356876f,-0.452733f,0.113603f,-0.058629f,-0.691430f,0.454745f,-0.828660f,-0.241287f,0.273692f,0.039244f,-0.273090f,0.001247f,0.261315f,1.092661f,0.498206f,-0.581792f,-0.119296f,0.079162f,0.970791f,-0.006329f,-0.449492f,0.031327f,-0.305933f,0.343043f,0.126759f,0.157613f,-0.302149f,0.342158f,-0.376675f,0.893763f,0.843673f,-0.185997f,-0.343288f,-0.814469f,0.744005f,-0.894441f,0.233085f};
float bias_data_dense_2[8] = {-0.040613f,-0.731753f,-0.394959f,-0.277660f,1.137739f,0.869172f,-0.543369f,-0.261291f};
float weights_data_dense_3[24] = {-0.126435f,1.127435f,0.138975f,-0.227646f,0.221582f,0.075386f,-0.436518f,-0.337974f,-0.013695f,-0.208857f,0.529911f,-0.570736f,-0.104903f,-0.473285f,-0.369365f,0.152757f,0.375481f,0.128993f,-0.428781f,-0.204452f,0.399355f,-0.314020f,0.163826f,-0.309669f};
float bias_data_dense_3[3] = {-0.315187f,1.356486f,-1.106717f};

void  setup() {
    Serial.begin(9600);
    Serial.setTimeout(50);
    while (!Serial) { ; }

    pinMode(s0,OUTPUT);
    pinMode(s1,OUTPUT);
    pinMode(s2,OUTPUT);
    pinMode(s3,OUTPUT);
    pinMode(out,INPUT);
   
    digitalWrite(s0,HIGH);  //Putting S0/S1 on HIGH/HIGH levels means the output frequency scalling is at 100%  (recommended)
    digitalWrite(s1,LOW);   //LOW/LOW is off HIGH/LOW is 20% and  LOW/HIGH is  2%
}

void loop() {
    uint16_t input_shape[] = {1, 3};
    uint16_t output_shape[] = {1, 3};

    float input_data[1 * 3];
    float output_data[1 * 3];

    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    // Neural Network Design
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(3, input_shape);
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_M(16, weights_data_dense_1, bias_data_dense_1);
    ailayer_relu_f32_t relu_layer_1 = AILAYER_SIGMOID_F32_M();
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_M(8, weights_data_dense_2, bias_data_dense_2);
    ailayer_relu_f32_t relu_layer_2 = AILAYER_RELU_F32_M();
    ailayer_dense_f32_t dense_layer_3 = AILAYER_DENSE_F32_M(3, weights_data_dense_3, bias_data_dense_3);
    ailayer_softmax_f32_t softmax_layer_3 = AILAYER_SOFTMAX_F32_M();

    ailayer_t *x;
    aimodel_t model;

    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_relu_f32_default(&relu_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_relu_f32_default(&relu_layer_2, x);
    x = ailayer_dense_f32_default(&dense_layer_3, x);
    x = ailayer_softmax_f32_default(&softmax_layer_3, x);
    model.output_layer = x;

    aialgo_compile_model(&model);

    uint32_t memory_size = aialgo_sizeof_inference_memory(&model);
    void *memory_ptr = (void *) malloc(memory_size);
    if (memory_ptr == NULL) {
        return 0;
    }

    aialgo_schedule_inference_memory(&model, memory_ptr, memory_size);

    while(1){
        //S2/S3  levels define which set of photodiodes we are using LOW/LOW is for RED LOW/HIGH  is for Blue and HIGH/HIGH is for green
        digitalWrite(s2,LOW);
        digitalWrite(s3,LOW);
        input_data[0] = pulseIn(out,LOW);
        Serial.print("Red = "); Serial.print(input_data[0]);

        digitalWrite(s2,LOW);
        digitalWrite(s3,HIGH);
        input_data[1] = pulseIn(out,LOW);
        Serial.print(" Blue = "); Serial.print(input_data[1]);

        digitalWrite(s2,HIGH);
        digitalWrite(s3,HIGH);
        input_data[2] = pulseIn(out,LOW);
        Serial.print(" Green = "); Serial.println(input_data[2]);

        aialgo_inference_model(&model, &input_tensor, &output_tensor);

        Serial.print("Max is: ");
        switch(array_max(output_data, 3)){
          case 0: Serial.println("Red"); break;
          case 1: Serial.println("Blue"); break;
          case 2: default: Serial.println("Green"); break;
        }
        delay(3000);
    }

    while(1){delay(5000);}
}