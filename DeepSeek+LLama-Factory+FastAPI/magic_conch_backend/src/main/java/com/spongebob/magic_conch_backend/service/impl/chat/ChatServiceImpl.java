package com.spongebob.magic_conch_backend.service.impl.chat;

import com.spongebob.magic_conch_backend.config.AiServiceConfig;
import com.spongebob.magic_conch_backend.service.ChatService;
import com.spongebob.magic_conch_backend.service.impl.chat.model.GenerateResponse;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.client.RestTemplate;

@Service
public class ChatServiceImpl implements ChatService {
    
    @Autowired
    private RestTemplate restTemplate;
    
    @Autowired
    private AiServiceConfig aiServiceConfig;

    @Override
    public String callAiForOneReply(String prompt) {
        // 获取基础URL http://localhost:8000
        String baseUrl = aiServiceConfig.getBaseUrl();
        // 构建完整的请求URL http://localhost:8000/generate?prompt=XXX
        String url = String.format("%s/generate?prompt=%s", baseUrl, prompt);
        // 发送GET请求并获取响应
        GenerateResponse response = restTemplate.getForObject(url, GenerateResponse.class);
        // 从响应中取出 generated_text 字段值返回
        return response != null ? response.getGenerated_text() : "";
    }
}
