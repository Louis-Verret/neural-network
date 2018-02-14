#ifndef PROGRESSBAR
#define PROGRESSBAR

#include <iostream>

class ProgressBar {
public:
    ProgressBar();
    ProgressBar(int size);
    ~ProgressBar();

    void display(float progress, float batch_error, float batch_metric);

protected:
    int m_size;
};

#endif
