from torch.utils.tensorboard import SummaryWriter


class EpochSummaryWriter(SummaryWriter):

    def __init__(self, *args, **kwargs):
        super(EpochSummaryWriter, self).__init__(*args, **kwargs)
        self.scalar_data = {}

    def update_scalar(self, tag, scalar_value):
        if tag not in self.scalar_data:
            self.scalar_data[tag] = {'value': 0.0, 'count': 0}
        self.scalar_data[tag]['value'] += scalar_value
        self.scalar_data[tag]['count'] += 1

    def add_update_scalars(self, *args, tag_suffix=None, **kwargs):
        for tag, data in self.scalar_data.items():
            average_value = data['value']/data['count'] if data['count'] > 0 else 0
            tag = f'{tag}/{tag_suffix}' if tag_suffix else tag
            super(EpochSummaryWriter, self).add_scalar(tag, average_value, *args, **kwargs)
        self.scalar_data = {}
