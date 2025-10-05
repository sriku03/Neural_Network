//以下は2入力，1出力の2層ニューラルネットワークである

#include <stdio.h>
#include <stdlib.h> //rand関数を使うために必要
#include <time.h>
#include <math.h>

#define H 2 // 隠れ総ユニットの数(隠れ層は1層でその層に2つのニューロンがある設定)

float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));  
}


float d_sigmoid_from_y(float y){
    return y * (1.0f - y);
}


int main(void){

    srand((unsigned)time(NULL)); // time()は<time.h>にある関数，関数の引数にNULLを渡すと結果だけ返してほしいという意味
    // 学習データ(AND)
    float X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; // 配列の名前の順番が決められるからこういう書き方
    float Y[4] = {0, 0, 0, 1};

    // パラメータ(学習して変えたいもの=重みとバイアス)

    float w1[2][H], b1[H], w2[H], b2;
    for (int j = 0; j < H; j++)
    {
        w1[0][j] = ((float)rand() / RAND_MAX - 0.5f);
        w1[1][j] = ((float)rand() / RAND_MAX - 0.5f);
        b1[j] = 0.0f; // 重みは乱数じゃなくていいみたい
        w2[j] = ((float)rand() / RAND_MAX - 0.5f);
    }
    b2 = 0.0f;

    float lr = 0.00001; // Irは学習率（どれくらいの大きさで係数を変化させるか）
    int E = 5000;          // 学習する回数
    for (int epoch = 0; epoch < E; epoch++)
    {
        float mse = 0.0f;
        for (int i = 0; i < 4; i++)
        { // 入力の組それぞれを入れている
            float x1 = X[i][0], x2 = X[i][1], y = Y[i];

            float z1[H];
            for (int j = 0; j < H; j++)
            { // ↑特定の組に対して，すべての２層目の値をパラメータから求めている
                float a = x1 * w1[0][j] + x2 * w1[1][j] + b1[j];
                z1[j] = sigmoid(a);
            }
            float a2 = b2; // ここから出力層を求めるところ
            for (int j = 0; j < H; j++)
            {
                a2 += z1[j] * w2[j];
            }
            float yhat = sigmoid(a2);

            float err = yhat - y;
            mse += 0.5f * err * err;
            // ここから各パラメータの偏微分を求めていく
            // まず出力層から//
            float delta2 = err * d_sigmoid_from_y(yhat);
            float delta1[H]; // ← これが必要！
            // 次隠れ層

            for (int j = 0; j < H; j++)
            {
                delta1[j] = (delta2 * w2[j]) * d_sigmoid_from_y(z1[j]);
            }
            // ここから上の二つを使って重みとバイアスによる微分を求めている
            for (int j = 0; j < H; j++)
            {
                w2[j] -= lr * delta2 * z1[j];
            }
            b2 -= lr * delta2;

            for (int j = 0; j < H; j++)
            {
                w1[0][j] -= lr * delta1[j] * x1;
                w1[1][j] -= lr * delta1[j] * x2;

                b1[j] -= lr * delta1[j];
            }
        }
        if (epoch % 500 == 0)
        {
            printf("epochは %4d  MSE=%.6f\n", epoch, mse / 4.0f);
        }
    }
    printf("\nPredictions: \n");
    // printf("二つの値を入力してください(空白を開けて)");
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int s = i;
            int t = j;

            float o, p, q, r;
            float u;
            float result;

            o = s * w1[0][0] + t * w1[1][0] + b1[0];
            p = sigmoid(o);
            q = s * w1[0][1] + t * w1[1][1] + b1[1];
            r = sigmoid(q);

            u = p * w2[0] + r * w2[1] + b2;
            result = sigmoid(u);

            printf("s=%d t=%d \n", s, t);

            if (result > 0.5f)
            {
                printf("結果は%dです \n", 1);
            }
            else
            {
                printf("結果は%dです \n", 0);
            }
        }
    }
}

