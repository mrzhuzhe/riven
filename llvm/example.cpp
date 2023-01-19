//  https://llvm.org/docs/GettingStarted.html
//  clang example.cpp -emit-llvm -S -o example2.ll why this doesnot equal to example.ll
#include <stdbool.h>

int G, H;
// _Bool what is this
int test(_Bool Condition) {
  int X;
  if (Condition)
    X = G;
  else
    X = H;
  return X;
}