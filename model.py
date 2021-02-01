import torch
import torch.nn as nn
import torchvision.models as models

class SAT(nn.Module):
    def __init__(self, encoder, decoder):
        super(SAT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, tgt, tgt_len, training):
        return self.decode(self.encode(x), tgt, tgt_len, training)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, annotation, tgt, tgt_len, training):
        return self.decoder(annotation, tgt.long(), tgt_len, training)

class Encoder(nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.layers = encoder
        self.fine_tune(False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 3, 1)
        return x # (batch_size, 14, 14, 512)

    def fine_tune(self, fine_tune=False):
        for p in self.layers.parameters():
            p.requires_grad = fine_tune

class Decoder(nn.Module):
    def __init__(self, D, L, decoder_dim, attention_dim, 
                 embedding_dim, vocab_size, pad_idx, sos_idx, eos_idx, device):
        super(Decoder, self).__init__()
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.D = D # Encoder dim
        self.L = L # Num pixels
        self.vocab_size = vocab_size
        self.init_m = nn.Linear(D, decoder_dim) # TODO: Init weight?
        self.init_h = nn.Linear(D, decoder_dim)
        self.attention = Attention(D, decoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(embedding_dim+D, decoder_dim, bias=True)
        
        self.L_o = nn.Linear(embedding_dim, vocab_size)
        self.L_h = nn.Linear(decoder_dim, embedding_dim)
        self.L_z = nn.Linear(D, embedding_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

        self.debug = nn.Linear(decoder_dim, D)
        self.sigmoid = nn.Sigmoid()
        
        self.embedding_layer = nn.Embedding(
                                            num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim,
                                            padding_idx=pad_idx)
        self.init_weights()

    def init_weights(self):
        self.embedding_layer.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, annotation, tgt, tgt_len, training, inference=False):
        """
        annotation: (batch_size, num pixels, encoder_dim)
        tgt: (batch_size, length)
        tgt_len: (batch_size)
        """
        # L = 196, D = 512
        # (batch_size, length, embedding_dim)
        tgt = self.embedding_layer(tgt)
        annotation = annotation.reshape(-1, self.L, self.D)
        memory = self.init_m(annotation.mean(dim=1)) # (batch_size, decoder_dim)
        hidden = self.init_h(annotation.mean(dim=1)) # (batch_size, decoder_dim)
        tgt_len = [l-1 for l in tgt_len]
        batch_size = tgt.shape[0]

        predictions = torch.zeros(batch_size, max(tgt_len) + 1, self.vocab_size).to(self.device)

        predictions[:, 0, :] = torch.zeros(self.vocab_size)\
                                .index_fill_(0, torch.tensor(self.sos_idx), torch.tensor(1))\
                                .to(self.device)
        alphas = torch.zeros(batch_size, max(tgt_len) + 1, self.L).to(self.device)

        for t in range(max(tgt_len)):
            batch_limit = sum([l > t for l in tgt_len])

            # (batch_size, encoder_dim), (batch_size, num_pixels)
            att_out, alpha = self.attention(annotation[:batch_limit], hidden[:batch_limit])
            
            # TODO: Is it okay?
            gate = self.sigmoid(self.debug(hidden[:batch_limit]))
            att_out = gate * att_out

            # (batch_size, embedding_dim + encoder_dim, decoder_dim)
            if training:
                lstm_input = torch.cat([tgt[:batch_limit, t, :], att_out], dim=1)
            else:
                _, pred = torch.max(predictions[:batch_limit, t, :], dim=1)
                lstm_input = torch.cat([self.embedding_layer(pred), att_out], dim=1)
            # (batch_size, decoder_dim), (batch_size, decoder_dim)
            hidden, memory = self.lstm(lstm_input, (hidden[:batch_limit], memory[:batch_limit]))


            # preds = torch.exp(self.L_o(self.E(predictions[:batch_limit, t, :]) + 
            #                             self.L_h(hidden) + 
            #                             self.L_z(att_out)))
            # preds = torch.exp(self.L_o(tgt[:batch_limit, t, :] + 
            #                             self.L_h(hidden) + 
            #                             self.L_z(att_out)))

            preds = self.fc(self.dropout(hidden))
            predictions[:batch_limit, t+1, :] = preds
            alphas[:batch_limit, t+1, :] = alpha.squeeze()
            if inference: # Inference
                _, pred = torch.max(preds, dim=1)
                if pred == torch.tensor(self.eos_idx):
                    return predictions, torch.tensor(t+2).view(1), alphas
        
        return predictions, tgt_len, alphas

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.annotation_att = nn.Linear(encoder_dim, attention_dim)
        self.hidden_att = nn.Linear(decoder_dim, attention_dim)
        self.att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, annotation, decoder_hidden):
        """
        annotation: output of encoder (batch_size, num pixels, encoder dim)
        decoder_hidden: (batch_size, deocder dim)
        return: weighted annotation (batch_size, attention_dim)
        """
        
        att_annotation = self.annotation_att(annotation)
        att_decoder_hidden = self.hidden_att(decoder_hidden).unsqueeze(1)
        # (batch_size, num pixels)
        m = self.att(self.tanh(att_annotation + att_decoder_hidden))
        # (batch_size, num pixels)
        alpha = self.softmax(m).squeeze(2)
        # (batch_size, encoder_dim)
        output = (annotation * alpha.unsqueeze(2)).sum(dim=1) 
        
        return output, alpha

def flat_model(model, layers):
    for layer in model.children():
        if type(layer) == nn.Sequential:
            flat_model(layer, layers)
        if list(layer.children()) == []:
            layers.append(layer)

def make_model(vocab, embedding_dim, pad_idx, sos_idx, eos_idx, device):
    encoder_model = models.vgg16(pretrained=True)
    encoder_layers = []
    flat_model(encoder_model, encoder_layers)
    model = SAT(
        Encoder(nn.ModuleList(encoder_layers[:30])),
        Decoder(512, 196, 512, 512, embedding_dim, vocab, pad_idx, sos_idx, eos_idx, device)
    )
    
    return model
