# 微服务(MICROSERVICES)与单体应用（MONOLITY）
## 简介和误区
单体应用一般指整个是一个很大的服务，服务内部都是进行函数调用。而微服务架构指将整个服务拆封成多个小服务，服务之间通过RPC（remote procedure call，远程过程调用）进行相互联系。每个微服务可以看作一个工作单元，包含独立的函数和数据库。

- 误区
  - 单体应用一般会被认为部属在单台机器上，而微服务是部属在多台机器上，这是不对的，单体应用也是具有扩展性，也能部署在多台服务器上。

## 单体应用优缺点
### 优点
#### 内部调用更快
相比网络请求，内部调用延时更低。
#### 更少重复代码
所有初始化配置等等操作，都可以共用，重复代码更少。
### 缺点
#### 需要了解更多的上下文
团队新入职成员时，需要更多时间去熟悉，因为，他需要了解整个服务的逻辑。
#### 复杂的部署
当修改应用的某个部分，整个服务需要全部重新部署。
#### 单节点故障
服务的一个环节出现问题，整个服务就会失败。

## 微服务优缺点
### 优点
#### 可扩展性
比单体应用更容易扩展。
#### 对新成员更友好
只需要了解自己需要编写逻辑的部分。
#### 更容易并行化工作
每个人可以专注于自己的服务，不像单体应用，可能存在依赖关系。
#### 更容易排查问题
一个地方出错可以很快定位到对应服务。
### 缺点
#### 难以设计
如何拆分服务，是微服务设计的难点。一个准则是，没有单一依赖的服务，如果两个服务是单一依赖的，则可进行合并。


