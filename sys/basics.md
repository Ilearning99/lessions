# System Design Basics

## Horizontal vs. Vertical Scaling
假设你有一台电脑，这台电脑，你在上面写代码，所以，你的电脑上正在运行一些程序。这个代码和正常的函数一样，他会接收一些输入，给出一些输出。其他人觉得你的代码非常有用，他们准备付款使用你的代码，这时，你不可能把你的电脑给所有人，你能做的事，通过某些协议，让你的代码在互联网上运行。你将你的代码通过api(应用程序接口)，将代码暴露出去，如果你的代码真正在运行，程序不是将输出写文件，或者，写数据库，而是，直接返回结果，结果叫做response，发出的东西叫做request。

初始化这个计算机，可能需要给它连一个数据库，可能是直接连在你的台式机上。需要配置endpoints，这样，用户才能连接你的计算机。你需要考虑，如果断电了，你应该怎么应对，你无法承受你的服务挂了，因为，别人给你付钱了。你应该把你的服务，配置在云端。云端和本地台式机有什么区别呢，没有区别，云端是一系列机器，别人提供给你的，你可以在上面运行算法。放在云端的好处事，云服务商可以保证机器的可靠性。这时，我们可以专注在我们自己的业务上。

随着业务发展，会有越来越多的请求，这个时候的问题，你现在的程序，已经无法处理所有的连接了。有两种解决方案，一种是买更多的机器，一种是买更好的机器。这个问题就是扩展能力。买更好的机器，就是vertical scaling。买更多的机器，horizontal scaling。 这是可以解决你系统扩展性的两种方法。

### 比较

Horizontal vs  Vertical

- Load balancing （负载均衡) vs. N/A
- Resilient (可恢复) vs. Single Point of failure
- Network call (RPC remote procedure calls, between two services) vs. (win) Inter process communication
- Data Inconsistency vs.(win) Consistent
- (win) Scales well as users increase vs. Hardware limit. 


### 系统设计的重点

- 可扩展性
- 可恢复
- 数据一致性

实际中有很多的权衡

