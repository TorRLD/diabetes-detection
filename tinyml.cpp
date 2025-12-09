#include <cstdint> // Adicione isto para corrigir o uint8_t
#include <stdio.h> // <--- Adicionado para funcionar o printf
#include "tinyml.h"
#include "model_data.h"

// --- MUDANÇA IMPORTANTE AQUI ---
// Em vez de all_ops_resolver, usamos o mutable (manual)
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Variáveis Globais
const int kTensorArenaSize = 6 * 1024;
// Alinhamento de memória para evitar erros no Pico
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

extern "C" int iniciar_modelo(void) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return -1;
    }

    // --- AQUI ESTÁ A CORREÇÃO DO ERRO ---
    // Criamos um resolvedor manual com espaço para 10 operações diferentes.
    static tflite::MicroMutableOpResolver<10> resolver;
    
    // Adicionamos apenas o que seu modelo precisa (baseado no seu hex dump)
    resolver.AddFullyConnected(); // Para as camadas 'Dense'
    resolver.AddRelu();           // Para ativação 'Relu'
    resolver.AddSoftmax();        // Geralmente usado na saída (probabilidade)
    resolver.AddQuantize();       // Caso o modelo seja quantizado
    resolver.AddDequantize();     // Caso o modelo seja quantizado
    resolver.AddReshape();        // Útil para garantir formatos
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return -2;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // --- LINHAS DE DEBUG ADICIONADAS ---
    // Isso vai aparecer no seu monitor serial assim que o Pico iniciar
    printf("\n[DEBUG] O modelo espera %d entradas.\n", input->dims->data[1]);
    printf("[DEBUG] O modelo entrega %d saidas.\n", output->dims->data[1]);
    // -----------------------------------

    return 0;
}

extern "C" float fazer_predicao(float* dados_entrada) {
    if (interpreter == nullptr) return 0.0f;

    // Proteção e Cópia dos dados
    int tamanho_esperado = input->dims->data[1]; // Deve ser 8
    for (int i = 0; i < tamanho_esperado; i++) {
        input->data.f[i] = dados_entrada[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return -1.0f;
    }

    // --- CORREÇÃO AQUI ---
    // O modelo tem 2 saídas: [Prob_Negativo, Prob_Positivo]
    // Queremos a Probabilidade Positiva, que fica no índice 1.
    float probabilidade_positivo = output->data.f[1];
    
    return probabilidade_positivo;
}