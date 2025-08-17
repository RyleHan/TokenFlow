# 订单管理服务 (Order Management Server)

## 服务概述
订单管理服务提供完整的订单生命周期管理，包括订单创建、查询、更新和取消等功能。

## API接口

### 1. 创建订单
**接口**: `create_order`
**功能**: 创建新订单
**参数**:
- `customer_id` (string): 客户ID
- `products` (array): 商品列表，包含商品ID和数量
- `shipping_address` (object): 配送地址信息
- `payment_method` (string): 支付方式

**返回值**:
- `order_id` (string): 订单ID
- `status` (string): 订单状态
- `total_amount` (float): 订单总金额
- `estimated_delivery` (string): 预计送达时间

### 2. 查询订单状态
**接口**: `get_order_status`
**功能**: 查询指定订单的当前状态
**参数**:
- `order_id` (string): 订单ID

**返回值**:
- `order_id` (string): 订单ID
- `status` (string): 订单状态 (pending/processing/shipped/delivered/cancelled)
- `tracking_number` (string): 物流追踪号
- `last_updated` (string): 最后更新时间

### 3. 更新订单
**接口**: `update_order`
**功能**: 更新订单信息
**参数**:
- `order_id` (string): 订单ID
- `updates` (object): 需要更新的字段

### 4. 取消订单
**接口**: `cancel_order`
**功能**: 取消指定订单
**参数**:
- `order_id` (string): 订单ID
- `reason` (string): 取消原因

## 状态码说明
- `pending`: 待处理
- `processing`: 处理中
- `shipped`: 已发货
- `delivered`: 已送达
- `cancelled`: 已取消

## 使用场景
- 电商平台订单管理
- B2B采购订单处理
- 餐饮外卖订单系统
- 服务预约订单管理