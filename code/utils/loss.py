import torch

def log_square_loss(output, label):
    output = output.squeeze()
    label = label.squeeze()
    return torch.mean((torch.log(torch.clamp(output, 0, 450) + 1) - torch.log(torch.clamp(label, 0, 450) + 1)) ** 2)


def log_distance_accuracy_function(output, label):
    # 128：batch size
    # 450应该是最大刑期37年，将outputs限幅到0~37年
    outputs = output.squeeze()
    label = label.squeeze()
    return float(torch.mean(torch.log(torch.abs(torch.clamp(outputs, 0, 450) - torch.clamp(label, 0, 450)) + 1)))


def acc25(output, label):
    output = output.squeeze()
    output = output.round()
    label = label.squeeze()
    return int(torch.sum(torch.abs(output-label)/label < 0.25))/len(output)
