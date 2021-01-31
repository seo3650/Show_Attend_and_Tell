import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from data.dataloader import DataSetWrapper
from model import make_model

import time
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from bleu import bleu_score

from PIL import Image
import skimage
import skimage.transform

def plot_loss_curve(train_loss_list, valid_loss_list, bleu4):
    x_axis = np.arange(len(train_loss_list))
    plt.figure()
    plt.plot(x_axis, train_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig(os.path.join('./result', 'Train_loss.png'))

    plt.clf()
    plt.figure()
    plt.plot(x_axis, valid_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Valid loss')
    plt.savefig(os.path.join('./result', 'Valid_loss.png'))

    plt.clf()
    plt.figure()
    plt.plot(x_axis, bleu4)
    plt.xlabel('epochs')
    plt.ylabel('BLEU')
    plt.savefig(os.path.join('./result', 'BLEU.png'))


def main(args):
    # Folder setting
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    if not os.path.isdir('./result'):
        os.mkdir('./result')

    # Hyparameters setting
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    dataset = DataSetWrapper(batch_size, args.num_workers, args.valid_size, input_shape=(224, 224, 3))
    train_loader, valid_loader = dataset.get_data_loaders()
    vocab = dataset.dataset.vocab
    
    model = make_model(len(vocab), 512, vocab.stoi['<PAD>'], vocab.stoi['<SOS>'], vocab.stoi['<EOS>'], device).to(device)
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # torch.set_printoptions(profile='full')
    # torch.autograd.set_detect_anomaly(True)
    print("Using device: " + str(device))

    if args.prev_model != '':
        checkpoint = torch.load('./model/' + args.prev_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # model.eval()

    if not args.inference:
        # Train
        best_bleu = 0
        train_loss_list, valid_loss_list = [], []
        bleu_list = []

        print('Start training')
        for epoch in range(epochs):
            train_total_loss, valid_total_loss = 0.0, 0.0
            model.train()
            for x, tgt, tgt_len, _ in train_loader:
                # tgt: (batch_size, tgt_len)
                # (batch_size, tgt_len, vocab_size), (batch_size, tgt_len, num pixels)
                x = x.to(device)
                tgt = tgt.to(device)
                tgt_len = tgt_len.to(device)
                predictions, tgt_len, alpha = model(x, tgt, tgt_len, True)

                predictions = predictions[:, 1:]
                packed_predictions = pack_padded_sequence(predictions, tgt_len, batch_first=True)[0].to(device)
                tgt = tgt[:, 1:]
                packed_tgt = pack_padded_sequence(tgt, tgt_len, batch_first=True)[0].to(device)

                optimizer.zero_grad()
                loss = criterion(packed_predictions, packed_tgt)
                loss += ((1 - alpha.sum(dim=1)) ** 2).mean()
                loss.backward()
                optimizer.step()

                _, predictions = torch.max(predictions, dim=2)

                train_total_loss += loss

            hypotheses = []
            references = []
            model.eval()

            with torch.no_grad():
                for x, tgt, tgt_len, all_tgt in valid_loader:
                    x = x.to(device)
                    tgt = tgt.to(device)
                    tgt_len = tgt_len.to(device)
                    predictions, tgt_len, alpha = model(x, tgt, tgt_len, False)

                    predictions = predictions[:, 1:]
                    packed_predictions = pack_padded_sequence(predictions, tgt_len, batch_first=True)[0].to(device)
                    tgt = tgt[:, 1:]
                    packed_tgt = pack_padded_sequence(tgt, tgt_len, batch_first=True)[0].to(device)

                    loss = criterion(packed_predictions, packed_tgt)
                    loss += ((1 - alpha.sum(dim=1)) ** 2).mean()

                    valid_total_loss += loss

                    _, predictions = torch.max(predictions, dim=2)

                    # Calculate BLEU
                    # TODO: Collect all reference captions, not one
                    predictions = predictions.cpu().tolist()
                    all_tgt = all_tgt.tolist()
                    t_prediciotns = []
                    t_tgt = []
                    for i in range(len(tgt)):
                        t_tgt.append(all_tgt[i])
                        t_prediciotns.append(predictions[i][:tgt_len[i] - 1])
                    predictions = t_prediciotns
                    tgt = t_tgt

                    hypotheses.extend(predictions)
                    references.extend(tgt)
                    assert len(references) == len(hypotheses)

            bleus = bleu_score(hypotheses, references)
            
            bleu_list.append(bleus[0])
            train_loss_list.append(train_total_loss)
            valid_loss_list.append(valid_loss_list)
            if (not args.no_save) and best_bleu <= bleus[0]:
                best_bleu = bleus[0]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'bleu': best_bleu
                }, './model/' + str(epochs) + ".pt")
                print('Model saved')

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +  
                                "|| [" + str(epoch) + "/" + str(epochs) + 
                                "], train_loss = " + str(train_total_loss) + 
                                ", valid_loss = " + str(valid_total_loss) + 
                                ", BLEU = " + str(bleus[4]) + ', ' +
                                str(bleus[0]) + '/' + str(bleus[1]) + '/' +
                                str(bleus[2]) + '/' + str(bleus[3])
                                )
            
        plot_loss_curve(train_loss_list, valid_loss_list, bleus)
    else:
        print("Start Inference")
        """
        Show image with caption
        ref: https://github.com/AaronCCWong/Show-Attend-and-Tell
        """

        # Load model
        assert args.prev_model != '' and args.img_path != ''
        checkpoint = torch.load('./model/' + args.prev_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        model.eval()

        img = Image.open(args.img_path).convert('RGB')
        img = dataset.dataset.transform(img)
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)

        img_feature = model.encode(img)
        img_feature = img_feature.view(img_feature.shape[0], -1, img_feature.shape[-1])
        tgt_len = [torch.tensor(args.max_length)]
        tgt = torch.zeros(1, args.max_length)
        sentence, tgt_len, alpha = model.decode(img_feature, tgt, tgt_len, False)
        _, sentence = torch.max(sentence, dim=2)
        sentence = sentence.squeeze()
        sentence = sentence[:tgt_len[0]]

        sentence = vocab.interpret(sentence.tolist())
        print(sentence)

        img = Image.open(args.img_path)
        w, h = img.size
        if w > h:
            w = w * 256 / h
            h = 256
        else:
            h = h * 256 / w
            w = 256
        left = (w - 224) / 2
        top = (h - 224) / 2
        resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
        img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
        img = img.astype('float32') / 255

        num_words = len(sentence)
        w = np.round(np.sqrt(num_words))
        h = np.ceil(np.float32(num_words) / w)
        alpha = alpha.clone().detach().squeeze()

        plot_height = np.ceil((num_words + 3) / 4.0)
        ax1 = plt.subplot(4, plot_height, 1)
        plt.imshow(img)
        plt.axis('off')
        for idx in range(num_words):
            ax2 = plt.subplot(4, plot_height, idx + 2)
            label = sentence[idx]
            plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
            plt.text(0, 1, label, color='black', fontsize=13)
            plt.imshow(img)

            shape_size = 14
            alpha_img = skimage.transform.pyramid_expand(alpha[idx, :].reshape(shape_size, shape_size), upscale=16, sigma=20)
            plt.imshow(alpha_img, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')

        plt.show()








        

if __name__ == '__main__':
    print("TRY14")
    parser = argparse.ArgumentParser(description="SAT")
    parser.add_argument(
        '--epochs',
        type=int,
        default=10
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--prev_model',
        type=str,
        default=''
    )
    parser.add_argument(
        '--inference',
        action='store_true'
    )
    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_argument(
        '--no_save',
        action='store_true'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=30
    )
    args = parser.parse_args()

    main(args)