# 库存管理服务 (Inventory Management Server)

## 服务概述
库存管理服务提供商品库存的实时管理，包括库存查询、更新、预留和释放等功能。

## API接口

### 1. 查询库存
**接口**: `check_inventory`
**功能**: 查询商品库存信息
**参数**:
- `product_id` (string): 商品ID
- `warehouse_id` (string): 仓库ID（可选）

**返回值**:
- `product_id` (string): 商品ID
- `available_stock` (int): 可用库存数量
- `reserved_stock` (int): 预留库存数量
- `total_stock` (int): 总库存数量
- `last_updated` (string): 最后更新时间

### 2. 批量查询库存
**接口**: `batch_check_inventory`
**功能**: 批量查询多个商品库存
**参数**:
- `product_ids` (array): 商品ID列表
- `warehouse_id` (string): 仓库ID（可选）

### 3. 预留库存
**接口**: `reserve_inventory`
**功能**: 为订单预留库存
**参数**:
- `product_id` (string): 商品ID
- `quantity` (int): 预留数量
- `reservation_id` (string): 预留ID
- `expires_in` (int): 预留超时时间（秒）

**返回值**:
- `reservation_id` (string): 预留ID
- `reserved_quantity` (int): 实际预留数量
- `expires_at` (string): 预留过期时间

### 4. 释放库存
**接口**: `release_inventory`
**功能**: 释放预留的库存
**参数**:
- `reservation_id` (string): 预留ID

### 5. 扣减库存
**接口**: `deduct_inventory`
**功能**: 正式扣减库存（确认销售）
**参数**:
- `reservation_id` (string): 预留ID
- `actual_quantity` (int): 实际扣减数量

### 6. 增加库存
**接口**: `add_inventory`
**功能**: 增加商品库存
**参数**:
- `product_id` (string): 商品ID
- `quantity` (int): 增加数量
- `warehouse_id` (string): 仓库ID
- `reason` (string): 增加原因

### 7. 库存调拨
**接口**: `transfer_inventory`
**功能**: 在仓库间调拨库存
**参数**:
- `product_id` (string): 商品ID
- `quantity` (int): 调拨数量
- `from_warehouse` (string): 源仓库ID
- `to_warehouse` (string): 目标仓库ID

## 库存状态说明
- `available`: 可用库存
- `reserved`: 已预留
- `damaged`: 损坏商品
- `expired`: 过期商品
- `in_transit`: 调拨中

## 预警机制
- 低库存预警
- 零库存提醒
- 过期商品提醒
- 滞销商品识别

## 使用场景
- 电商库存管理
- 零售门店库存
- 仓储管理系统
- 供应链管理