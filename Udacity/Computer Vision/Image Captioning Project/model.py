import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_p=0.5):
        super(DecoderRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
            
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, hidden_size, num_layers, dropout=drop_p, batch_first=True)
        
        self.classifier = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions, hidden):
            
        #self.hidden = self.hidden.detach()
        # Embedding the captions
        embedded = self.embedding(captions[:,:-1])
        
        # concat features and capations to one tensor
        embedded = torch.cat((features.unsqueeze(1), embedded), dim=1)
        
        # Passing through
        lstm_out, hidden = self.lstm(embedded, hidden)
            
        # Classofy lstm_output
        out = self.classifier(lstm_out)
        
        return out, hidden
        

    def init_hidden_state(self, batch_size):
        
        weight = next(self.parameters()).data
        
        if torch.cuda.is_available():
            
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        
        return hidden


    def sample(self, inputs, states=None, max_len=20):
        
        # Initialize the hidden state
        if states is None:
            hidden = self.init_hidden_state(inputs.shape[0])
        else:
            hidden = states
        
        output_list = []
        
        # setting word counter
        word_len = 0
        
        with torch.no_grad():
            while word_len < max_len:
                lstm_out, hidden = self.lstm(inputs, hidden)
                out = self.classifier(lstm_out)
                
                # squeeze except of batch size
                out = out.squeeze(1)
                out = out.argmax(dim=1)

                output_list.append(out.item())
                inputs = self.embedding(out.unsqueeze(0))
                
                word_len += 1
                
                # if out is 1 = <end> break loop
                if out == 1:
                    break
        
        return output_list