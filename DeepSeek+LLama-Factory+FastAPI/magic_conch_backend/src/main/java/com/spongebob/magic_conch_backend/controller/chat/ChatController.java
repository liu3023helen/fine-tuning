package com.spongebob.magic_conch_backend.controller.chat;

import com.spongebob.magic_conch_backend.common.enums.ResultCode;
import com.spongebob.magic_conch_backend.common.vo.Result;
import com.spongebob.magic_conch_backend.service.ChatService;

import io.micrometer.common.util.StringUtils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping(value = "/chat")
public class ChatController {

    @Autowired
    private ChatService chatService;

    @RequestMapping("/generate")
    @ResponseBody
    public Result generate(@RequestParam String prompt) {
        Result result = Result.success();
        if(StringUtils.isBlank(prompt)) {
            return Result.error(ResultCode.PARAM_INVALID,"prompt不能为空");
        }
        try {
            String res = chatService.callAiForOneReply(prompt);
            result.setData(res);
        } catch (Exception e) {
            result = Result.error();
        }
        return result;
    }

}
