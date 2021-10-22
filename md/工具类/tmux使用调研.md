# Tmux科研利器-更方便地管理实验

## 1. 概念解释

基础部件是session(会话)

每个会话可以创建多个window(窗口)

每个窗口可以划分多个pane(窗格)


## 2. 常用命令解释

- tmux的退出 : `ctrl + d 或者 exit命令`

- tmux前缀键：Ctrl+b+x

&ensp;&ensp;&ensp;&ensp;- x=? 的时候是帮助信息（按q退出）

&ensp;&ensp;&ensp;&ensp;- x=d 分离当前session

&ensp;&ensp;&ensp;&ensp;- x=s 列出所有session

&ensp;&ensp;&ensp;&ensp;- x=$ 重命名session

&ensp;&ensp;&ensp;&ensp;- x=n 切换到下一个窗口

- 启动tmux session: `tmux new -s <session_name> `

- 分离session:` tmux detach (效果是退出当前tmux窗口)`

- 接入session: `tmux attach（重新接入tmux窗口`）【tmux at -t <session_name>】

- 关闭会话：`tmux ls | grep : | cut -d. -f1 | awk '{print substr($1, 0, length($1)-1)}' | xargs kill`

- 查看`tmux session: tmux ls `

- 杀死`tmux session: tmux kill-session -t <session_name>`

- 重命名`session: tmux rename-session -t 0 <new_name>`

- 切换`session: tmux switch -t <session_name>`

- 划分窗格(pane)：

&ensp;&ensp;&ensp;&ensp;- 上下：`tmux split-window`  或者 `ctrl+b+"`

&ensp;&ensp;&ensp;&ensp;- 左右：`tmux split-window -h` 或者 `ctrl+b+%`

- 选择pane: 

&ensp;&ensp;&ensp;&ensp;- 向上：`tmux select-pane -U` 或者 `ctrl+b+↑`

&ensp;&ensp;&ensp;&ensp;- 向下：`tmux  select-pane -D` 或者 `ctrl+b+↓` 

&ensp;&ensp;&ensp;&ensp;- 向左：`tmux select-pane -L` 或者 `ctrl+b+←`

&ensp;&ensp;&ensp;&ensp;- 向右：`tmux select-pane -R` 或者 `ctrl+b+→`


## 3. tmux工作流程说明

1 新建会话，起名要慎重，做实验的时候也要对得上号。

`tmux new -s repnas_trail1`

2 在新的窗口执行命令

`python train.py`

**如果需要在同一个窗口对比，可以划分pane** 

&ensp;&ensp;&ensp;&ensp;- `ctrl+b+"` 上下划分

&ensp;&ensp;&ensp;&ensp;- `ctrl+b+%`左右划分

- 想切换pane: `ctrl+b+（ctrl+方向键）`

- 想调整pane大小：`ctrl+b+(alt+方向键)`

- 想关闭pane: `ctrl+b+x`(kill) 或者 `ctrl+d` (后台运行)

- 想要翻页查看log: `ctrl+b+[` 可以使用pageup pagedown进行翻译，退出需要按q

3 退出tmux session

`tmux detach`：放在后台运行 快捷键：`ctrl+b+d`

`ctrl + d ` : 直接删除session

4 重新进入tmux session

查看目前的tmux session: `tmux ls`

`tmux attach -t <session_name>` 或者 `tmux a -t <session_name>`

