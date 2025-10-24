#include <stdio.h>
#include "aifes.h"

void print_tensor_like_array(const char* name, const aitensor_t *tensor) {
    uint32_t total = 1;
    for (uint32_t i = 0; i < tensor->dim; i++) {
        total *= tensor->shape[i];
    }

    printf("float %s[%u] = {", name, total);
    for (uint32_t i = 0; i < total; i++) {
        printf("%ff", ((float*)tensor->data)[i]);
        if (i < total - 1)
            printf(", ");
    }
    printf("};\n");
}

int main(int argc, char *argv[]) {
    // Dataset Preparation
    float input_data[4 * 2] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    float target_data[4 * 1] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };
    float output_data[4 * 1];

    uint16_t input_shape[] = {4, 2};
    uint16_t target_shape[] = {4, 1};
    uint16_t output_shape[] = {4, 1};
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);


    // Neural Network Design
    aimodel_t model;

    uint16_t input_layer_shape[] = {1, 2};
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(2, input_layer_shape);
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(2);
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_A();
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(1);
    ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_A();

    ailayer_t *x;

    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    model.output_layer = x;

    // Neural Network Compilation
    ailoss_mse_t mse_loss;
    model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);

    aialgo_compile_model(&model);

    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    void *parameter_memory = malloc(parameter_memory_size);
    if (parameter_memory == NULL) {
        return 0;
    }

    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    aimath_f32_default_init_glorot_uniform(&dense_layer_1.weights);
    aimath_f32_default_init_zeros(&dense_layer_1.bias);

    aimath_f32_default_init_glorot_uniform(&dense_layer_2.weights);
    aimath_f32_default_init_zeros(&dense_layer_2.bias);

    aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.1f, 0.9f, 0.999f, 1e-7);
    aiopti_t *optimizer = aiopti_adam_f32_default(&adam_opti);

    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    void *memory_ptr = malloc(memory_size);
    if (memory_ptr == NULL) {
        return 0;
    }

    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

    aialgo_init_model_for_training(&model, optimizer);

    //Training
    uint32_t batch_size = 4;
    uint16_t epochs = 100;

    printf("Start training\n");
    for (uint32_t i = 0; i < epochs; i++) {
        aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);
    }
    printf("Finished training\n");

    //Testing
    aialgo_inference_model(&model, &input_tensor, &output_tensor);

    for (uint32_t i = 0; i < 4; i++) {
        printf("%f\n", output_data[i]);
    }

    // Neural Network Export
    print_tensor_like_array("weights_data_dense_1", &(dense_layer_1.weights));
    print_tensor_like_array("bias_data_dense_1", &(dense_layer_1.bias));
    print_tensor_like_array("weights_data_dense_2", &(dense_layer_2.weights));
    print_tensor_like_array("bias_data_dense_2", &(dense_layer_2.bias));

    //Memory free
    free(parameter_memory);
    free(memory_ptr);
}
