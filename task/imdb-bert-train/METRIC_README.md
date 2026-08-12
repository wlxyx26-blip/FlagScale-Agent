
###评测FlagScale-Agent 端到端训练测试任务###

任务得分;
    task_score = task_success × task_quality

其中，
（1）task_score代表任务完成度，判断智能体能否完成任务，最终综合得分为1；否则为0：

task_success =  artifact_score and protocol_score and model_loadable and consistent and modei_quality_pass

--artifact_score: 要求生成的文件是否真实存在(0/1)  
--protocol_score: agent训练参数是否与符合要求(0/1) 
--model_loadable：Verifier能够在独立进程中加载模型（0/1）
--consistent：Agent报告结果与Verifier独立复算结果上下浮动最大不超任务预设的数值容差 暂定0.01（0/1）
--model_quality_pass:预设模型主指标应达到任务预先标定的质量合格线。暂定0.01（0/1）

与第一次运行时得出的主指标结果作为合格线0.9206

#合格线由参考方案多次独立运行的单侧95%预测边界与任务最低有效性能共同确定。

不同类型任务的主指标不同，本次训练任务以ACC为主指标

（2）task_quality代表任务质量，判断智能体完成任务的“有多好”，使用连续评分：
这部分使用LLM-as-judge作为评分机制，主要对轨迹以及agent生成的结果进行评分，将评价项的分数，按照各自的权重做加权平均，得到计算方式如下：
    task_quality = （wp * process + wr * robustness） / (wp + wr)
其中，当前wp = 1、wr = 1

 --process：执行过程质量-工具选择是否正确，传入参数是否合理，是否错误使用了无关工具，整体执行任务顺序是否合理  （0-100）
重点评价：
1. 执行过程中工具选择是否适合当前任务；
2. 工具调用传入参数，以及每个工具调用后有对应的观察结果；
3. 环境检查、数据检查、代码编写、训练、评估、模型保存和最终验证的顺序是否合理；
4. 是否存在明显无关、重复或没有产生新信息的操作；
5. 是否根据工具返回结果调整后续行动；
    
--robustness：解决问题能力-检查智能体遇到异常后是否能够有效恢复，比如命令执行失败后，有无分析原因，工具被阻止后，有没有更换方法等  
    如果轨迹没有出现异常评分直接是100；如果轨迹出现异常、阻断或者拒绝时，根据具体。
重点评价：
1. 是否可以识别工具、命令、环境、训练或文件异常；
2. 出现异常或阻断后，是否理解失败原，并且能否针对问题改变命令、参数、代码或执行方案；
3. 修复后是否重新验证；
4. 是否存在重复触发相同错误或未解决异常；
5. 多轮被被阻断后，最终执行状态是否稳定；
6.文件或者其余不存在，有没有重新确认路径。


###后续也可以将token消耗量，工具调用次数等纳入评分规则中，如果消耗过高，将评分拉下  调研，当前要融合大模型判定      
分析token消耗###

三、统计资源
在评估框架中，可以直接提取出运行任务的输入Token，输出Token，缓存Token，工具调用次数，调用工具数，LLM调用次数，LLM总延迟，总体运行时间等
path：./jobs/<id>/<task name>/agent/usage-summary.json

#这部分当前代码还得需要精细

###eg. 比较统一任务下效率###
resource_efficiency =
    0.25 × token_efficiency
  + 0.25 × llm_call_efficiency
  + 0.25 × tool_call_efficiency
  + 0.25 × runtime_efficiency

选取中位Token中位、工具调用次数、中位LLM调用次数、中位运行时间




