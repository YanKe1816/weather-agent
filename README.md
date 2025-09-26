# weather-agent
一个测试用的天气 Agent，使用 OpenAI Agent SDK 的接口配置一个最小的天气查询工具，
内部使用模拟数据以便在无网络环境中也能运行。

## 快速开始

```bash
pip install -r requirements.txt
python -m weather_agent 北京
```

如未配置 `OPENAI_API_KEY` 或当前环境无法访问 OpenAI 接口，示例会自动回退到本地模拟
数据，因此仍能得到确定性的天气字符串。

## Agent 结构

`weather_agent` 模块中的 `query_weather` 函数展示了如何通过 OpenAI Agent SDK 创建 Agent、
注册 `get_mock_weather` 工具，并在需要时把工具调用映射到本地的模拟天气数据。
