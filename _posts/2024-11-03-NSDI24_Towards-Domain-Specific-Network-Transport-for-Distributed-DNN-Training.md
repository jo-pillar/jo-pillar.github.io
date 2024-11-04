---
title: NSDI24:Towards Domain-Specific Network Transport for Distributed DNN Training
date: 2024-11-03 23:38:27 +0800
categories: [论文笔记, NSDI]
tags: [dnn,protocol,chenkai,mlsys]     # TAG names should always be lowercase
---

# NSDI24:Towards Domain-Specific Network Transport for Distributed DNN Training

本文认为当前的网络传输协议没有充分利用DNN训练应用当中的丰富流量特性，于是提出了一个针对DNN训练的领域特定的网络传输协议MLT

## DNN训练具有哪些流量特性
•  包间独立性--->包级负载均衡

•  包与包亦有不同--->back layer的梯度更新 要比 forward layer的梯度更重要 大梯度要比小梯度重要-> 给包打标签来指导交换机如何进行优先级排队和选择性丢包

•  局部损失不影响大局,丢失一些梯度信息也不会影响收敛

•  最后一关由我来守:只有当丢包率超过了某个百分比后才会触发重传

## 如何进行包级负载均衡
如何应对包级均衡张量中的梯度信息被分割成独立群组，然后被包装成数据包带有tensor ID 层和位移信息，然后根据这些信息计算张量应该被放入的内存地址 使用源路由来进行负载均衡，没提路径是如何指定的，盲猜是将交换机侧的ecmp的思想挪到主机侧来使用，也就是随机指定路径。

Testbed的搭建

只要交换机端口够 一台可以当八台 ------- Each leaf switch has two 100Gbps links connecting to the spine switch,thus logically we have two spine switches.

我不知道的事

•  

In each iteration, each worker can send and receive model gradients with tens to thousands of MBs(一轮迭代 每个worker的通讯量高达10到千兆)

•  

在训练时 为了避免单次通信量过小 训练框架会将张量打包成bucket一起传输

Ps:写这个不是为了做论文翻译,所以我只写了我认为的论文的核心贡献,和一些我关心的地方,有不足之处请参阅原文.

