#include<iostream>

int main() {
    const int N = 10;
    int arr[N] = {5, 23, 23, 234, 54, 233, 23, 54, 65, 25};
    int max = 0;

    std::cout << "Array:";
    for (int i = 0; i < N; i++){
        std::cout << " " << arr[i];
        max = std::max(max, arr[i]);
    }
    std::cout << std::endl;
    std::cout << "The max is: " << max << std::endl;

}