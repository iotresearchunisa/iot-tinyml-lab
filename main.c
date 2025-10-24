#include <stdio.h>
#include "aifes.h"

float weights_data_dense_1[4] = {-6.452141f, -6.947195f, 6.637261f, 6.646863f};
float bias_data_dense_1[2] = {3.446402f, -3.613925f};
float weights_data_dense_2[2] = {-6.310305f, 6.392910f};
float bias_data_dense_2[1] = {3.095918f};

int main(int argc, char *argv[]) {
    uint16_t input_shape[] = {1, 2};
    uint16_t output_shape[2] = {1, 1};

    float input_data[] = {1.0f, 1.0f};
    float output_data[1 * 1];

    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    aimodel_t model;

    uint16_t input_layer_shape[] = {1, 2};
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(2, input_layer_shape);
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_M(2, weights_data_dense_1, bias_data_dense_1);
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_M();
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_M(1, weights_data_dense_2, bias_data_dense_2);
    ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_M();

    ailayer_t *x;
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    model.output_layer = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);

    aialgo_compile_model(&model);

    uint32_t memory_size = aialgo_sizeof_inference_memory(&model);
    void *memory_ptr = (void *) malloc(memory_size);
    if (memory_ptr == NULL) {
        return 0;
    }

    aialgo_schedule_inference_memory(&model, memory_ptr, memory_size);

    aialgo_inference_model(&model, &input_tensor, &output_tensor);

    printf("%f\n", ((float *) output_tensor.data)[0]);

    free(memory_ptr);
}
