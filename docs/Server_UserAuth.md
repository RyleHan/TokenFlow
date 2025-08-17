# 用户认证服务 (User Authentication Server)

## 服务概述
用户认证服务提供用户身份验证、授权管理和会话控制功能，支持多种认证方式。

## API接口

### 1. 用户登录
**接口**: `login`
**功能**: 用户身份验证登录
**参数**:
- `username` (string): 用户名或邮箱
- `password` (string): 密码
- `remember_me` (boolean): 是否记住登录状态

**返回值**:
- `access_token` (string): 访问令牌
- `refresh_token` (string): 刷新令牌
- `expires_in` (int): 令牌有效期（秒）
- `user_info` (object): 用户基本信息

### 2. 用户注册
**接口**: `register`
**功能**: 新用户注册
**参数**:
- `username` (string): 用户名
- `email` (string): 邮箱地址
- `password` (string): 密码
- `confirm_password` (string): 确认密码
- `profile` (object): 用户档案信息

**返回值**:
- `user_id` (string): 用户ID
- `activation_required` (boolean): 是否需要激活
- `message` (string): 注册结果消息

### 3. 用户登出
**接口**: `logout`
**功能**: 用户退出登录
**参数**:
- `access_token` (string): 访问令牌

### 4. 令牌刷新
**接口**: `refresh_token`
**功能**: 刷新访问令牌
**参数**:
- `refresh_token` (string): 刷新令牌

### 5. 验证令牌
**接口**: `validate_token`
**功能**: 验证令牌有效性
**参数**:
- `access_token` (string): 访问令牌

### 6. 重置密码
**接口**: `reset_password`
**功能**: 重置用户密码
**参数**:
- `email` (string): 用户邮箱
- `reset_token` (string): 重置令牌（可选）
- `new_password` (string): 新密码（可选）

## 安全特性
- JWT令牌认证
- 密码加密存储
- 会话管理
- 多重身份验证支持
- 防暴力破解机制

## 使用场景
- Web应用用户登录
- 移动应用身份验证
- API接口权限控制
- 单点登录(SSO)系统