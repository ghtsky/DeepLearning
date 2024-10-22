from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# writer.add_scalar()：在summary中添加标量数据

"""
Args:
            tag (str): Data identifier 标识符
            scalar_value (float or string/blobname): Value to save  Y轴 
            global_step (int): Global step value to record  X轴
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
              style (simple_value field). New style could lead to faster data loading.
"""


for i in range(100):
    writer.add_scalar("y=x", i, i)

for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

for i in range(100):
    writer.add_scalar("y=2x", 3 * i, i)

writer.close()