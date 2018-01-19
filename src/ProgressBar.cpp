#include "ProgressBar.h"

ProgressBar::ProgressBar() : m_size(0) {}

ProgressBar::~ProgressBar() {
}

ProgressBar::ProgressBar(int size) : m_size(size) {
}

void ProgressBar::display(float progress, float batch_error, float batch_metric) {
    int pos = m_size * progress;
    std::cout << "[";
    for (int i = 0; i < m_size; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] [" << int(progress * 100.0) << "% | Error: " << batch_error << " | Accuracy: " << batch_metric << "]\r";
    std::cout.flush();
}
