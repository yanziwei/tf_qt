# Overview
This repo is used for deep learning transformer training of Chinese commodity futures data; you need to download the futures data to the local;

**Features**

1. random sample the CVS tick data of the day;
2. The maximum sampling time is 30 minutes; The GT label is the ups and downs of 1 minute and 5 minutes.
3. The main network uses a transformer, and the output uses a full connection to map to a multi-classification of 2 labels(up or down).
4. The experiment found that the loss decreased very slowly. The accuracy rate of the verification set was 67% in 1min label, 40% in 5min label. I will further explore how to use deep learning to trade on market.