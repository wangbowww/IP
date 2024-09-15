#pragma once
#include <assert.h>
typedef unsigned char uchar;
/*
其中input, width, height, inStep, inChannels分别是输入图像的数据、宽、高、step和通道数。
Output和outStep是输出图像的数据和step，宽高与输入图像相同、通道数为1.
channelToGet是要获取的通道索引，比如如果input是BGR格式的数据，则channelToGet=0将获取B通道， channelToGet=1将获取G通道……
*/
void getChannel(const uchar* input, int width, int height, int inStep, int inChannels, uchar* output, int outStep, int channelToGet)
{
    assert(channelToGet >= 0 && channelToGet < inChannels); // 获取的通道有效
    // (x, y)表示像素位置，而每个像素由inChannels个通道组成，这里每个通道是一个uchar
    const uchar *row = input;
    uchar *outRow = output;
    for (int y = 0; y < height; ++y, row += inStep, outRow += outStep) {
        uchar *px = outRow;
        for(int x = 0; x < width; ++x, px += 1){
            *px = *(row + x * inChannels + channelToGet); 
        }
    }
}