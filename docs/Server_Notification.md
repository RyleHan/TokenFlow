# 通知服务 (Notification Server)

## 服务概述
通知服务提供多渠道消息通知功能，支持邮件、短信、推送通知等多种方式。

## API接口

### 1. 发送邮件通知
**接口**: `send_email`
**功能**: 发送邮件通知
**参数**:
- `recipient` (string): 收件人邮箱
- `subject` (string): 邮件主题
- `content` (string): 邮件内容
- `template_id` (string): 邮件模板ID（可选）
- `attachments` (array): 附件列表（可选）

**返回值**:
- `message_id` (string): 消息ID
- `status` (string): 发送状态
- `sent_at` (string): 发送时间

### 2. 发送短信通知
**接口**: `send_sms`
**功能**: 发送短信通知
**参数**:
- `phone_number` (string): 手机号码
- `message` (string): 短信内容
- `template_id` (string): 短信模板ID（可选）

### 3. 发送推送通知
**接口**: `send_push_notification`
**功能**: 发送移动推送通知
**参数**:
- `user_id` (string): 用户ID
- `title` (string): 通知标题
- `body` (string): 通知内容
- `data` (object): 附加数据
- `priority` (string): 优先级

### 4. 批量通知
**接口**: `send_bulk_notification`
**功能**: 批量发送通知
**参数**:
- `recipients` (array): 收件人列表
- `notification_type` (string): 通知类型
- `content` (object): 通知内容

### 5. 查询通知状态
**接口**: `get_notification_status`
**功能**: 查询通知发送状态
**参数**:
- `message_id` (string): 消息ID

**返回值**:
- `message_id` (string): 消息ID
- `status` (string): 状态
- `delivered_at` (string): 送达时间
- `read_at` (string): 阅读时间

### 6. 设置通知偏好
**接口**: `set_notification_preferences`
**功能**: 设置用户通知偏好
**参数**:
- `user_id` (string): 用户ID
- `preferences` (object): 偏好设置

## 通知类型
- `email`: 邮件通知
- `sms`: 短信通知
- `push`: 推送通知
- `in_app`: 应用内通知
- `webhook`: Webhook通知

## 状态说明
- `queued`: 已排队
- `sent`: 已发送
- `delivered`: 已送达
- `failed`: 发送失败
- `read`: 已读取

## 模板管理
- 邮件模板定制
- 短信模板管理
- 多语言支持
- 动态内容替换

## 使用场景
- 订单状态通知
- 系统告警提醒
- 营销推广消息
- 用户行为触发通知