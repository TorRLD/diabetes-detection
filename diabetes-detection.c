#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "tinyml.h" 

#define NUM_ENTRADAS 8 

// Função auxiliar para leitura segura
float ler_float_seguro() {
    char buffer[32];
    int index = 0;
    while (true) {
        int c = getchar_timeout_us(100000);
        if (c == PICO_ERROR_TIMEOUT) {
            if (index > 0) { printf("\n"); break; }
            continue;
        }
        if (c == '\n' || c == '\r') { printf("\n"); break; }
        if (c == '\b' || c == 127) { if (index > 0) { printf("\b \b"); index--; } }
        else if (index < 30) { buffer[index++] = (char)c; printf("%c", c); }
    }
    buffer[index] = '\0';
    return (float)atof(buffer);
}

int main() {
    stdio_init_all();
    sleep_ms(3000); 

    printf("\n========================================\n");
    printf("   SISTEMA DE VALIDACAO DE EFICACIA\n");
    printf("   Modo: 8 Variaveis (Sem Imputacao)\n");
    printf("========================================\n");

    if (iniciar_modelo() != 0) {
        printf("ERRO CRITICO: Falha ao carregar modelo!\n");
        return -1;
    }
    printf("Modelo carregado! Pronto.\n\n");

    float entradas[NUM_ENTRADAS];
    
    // Constantes de Normalização (MinMax Scaler do Dataset Pima)
    const float MIN_VALS[] = { 0.0f,   0.0f,   0.0f,  0.0f,   0.0f,  0.0f, 0.078f, 21.0f };
    const float MAX_VALS[] = { 17.0f, 199.0f, 122.0f, 99.0f, 846.0f, 67.1f, 2.42f,  81.0f };

    while (true) {
        printf("--- INSERIR DADOS DO PACIENTE ---\n");
        float val;

        // 1. Gravidezes
        printf("1. Numero de Gravidezes (0 se homem/nulo): ");
        val = ler_float_seguro();
        entradas[0] = (val - MIN_VALS[0]) / (MAX_VALS[0] - MIN_VALS[0]);

        // 2. Glicose
        printf("2. Glicose (mg/dL): ");
        val = ler_float_seguro();
        entradas[1] = (val - MIN_VALS[1]) / (MAX_VALS[1] - MIN_VALS[1]);

        // 3. Pressão
        printf("3. Pressao Diastolica (mm Hg): ");
        val = ler_float_seguro();
        entradas[2] = (val - MIN_VALS[2]) / (MAX_VALS[2] - MIN_VALS[2]);

        // 4. Espessura da Pele
        printf("4. Espessura da Pele (mm) [Media=20]: ");
        val = ler_float_seguro();
        entradas[3] = (val - MIN_VALS[3]) / (MAX_VALS[3] - MIN_VALS[3]);

        // 5. Insulina
        printf("5. Insulina (mu U/ml) [Media=80]: ");
        val = ler_float_seguro();
        entradas[4] = (val - MIN_VALS[4]) / (MAX_VALS[4] - MIN_VALS[4]);

        // 6. IMC
        printf("6. IMC (ex: 30.0): ");
        val = ler_float_seguro();
        entradas[5] = (val - MIN_VALS[5]) / (MAX_VALS[5] - MIN_VALS[5]);

        // 7. Histórico Familiar (Pedigree)
        // Dica: 0.08 = Sem histórico, 0.5 = Histórico moderado, >1.0 = Forte
        printf("7. Historico (0.08 a 2.42): ");
        val = ler_float_seguro();
        entradas[6] = (val - MIN_VALS[6]) / (MAX_VALS[6] - MIN_VALS[6]); 

        // 8. Idade
        printf("8. Idade: ");
        val = ler_float_seguro();
        entradas[7] = (val - MIN_VALS[7]) / (MAX_VALS[7] - MIN_VALS[7]);

        printf("Processando 8 variaveis...\n");

        float probabilidade = fazer_predicao(entradas);
        float porcentagem = probabilidade * 100.0f;

        printf("\n>>> RESULTADO IA: %.2f%%\n", porcentagem);

        // Limiar de decisão
        if (probabilidade > 0.50f) { // Com 8 dados reais, podemos baixar o corte para 50%
            printf(">>> DIAGNOSTICO: POSITIVO (Risco Detectado)\n");
        } else {
            printf(">>> DIAGNOSTICO: NEGATIVO (Normal)\n");
        }
        
        printf("----------------------------------------\n\n");
        sleep_ms(1000);
    }
    return 0;
}