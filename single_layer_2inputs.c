//以下は1入力，1出力の1層ニューラルネットワークである
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

float sigmoid(float x){
    return 1.0f/(1.0f+expf(-x));
}


int main(){
    srand((unsigned)(time(NULL) ^ clock()));
    float X[4][2] = {0,0,0,1,1,0,1,1};
    float Y[4] = {0,1,1,1};
    //上までの時点でw1,w2,b1による損失関数は決まっている

    //ここ以降損失関数の最小値を推測するために，ランダムな一点を取得して，そこから，勾配ベクトルを求めるという方向で進む
    float w1 = ((float)rand()/RAND_MAX - 0.5f);
    float w2 = ((float)rand()/RAND_MAX - 0.5f);
    float b1 = ((float)rand()/RAND_MAX - 0.5f);

    //損失関数の一点を得るところfor文使えば簡単
    for (int j=0; j<2000; j++){
        float z;
        float r;
        float loss = 0.0f;
        float gw1 = 0.0f;
        float gw2 = 0.0f;
        float gb1 = 0.0f;
        
        for(int i=0; i<4; i++){
            z = w1*X[i][0] + w2*X[i][1] + b1;
            r = sigmoid(z);
            //今回の損失関数としてはMSE(Mean squared Error)をつかうこれを最小にすることを目指す
            float e = r - Y[i];
            loss += e*e;

            gw1 += e*r*(1-r)*X[i][0];
            gw2 += e*r*(1-r)*X[i][1];
            gb1 += e*r*(1-r);
        }
        loss = loss/4.0f;
        gw1 /= 4.0f;
        gw2 /= 4.0f;
        gb1 /= 4.0f;



        // printf("MSEは%f\nw1は%f\nw2は%f\nb1は%f\n", loss, w1, w2, b1);
        // printf("gwは%f\ngw2は%f\ngb1は%f\n\n\n\n\n", gw1, gw2, gb1);

        float lr = 0.5f;
        w1 -= lr*gw1;
        w2 -= lr*gw2;
        b1 -= lr*gb1;
    }
    // printf("%f %f %f", w1, w2, b1);
    printf("2つの数字を入力してください(空白入れて)");
    int a,b;
    float result;

    scanf("%d %d", &a,&b);
    result = w1*a + w2*b + b1;

    if (result<0.5f){
        printf("結果は%d", 0);
    }
    else{
        printf("結果は%d", 1);
    }

}

