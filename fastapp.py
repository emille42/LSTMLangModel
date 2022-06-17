from models import TxtLSTM
import torch
from generators import GreedGenerator, BeamGenerator
import youtokentome as yttm
import json

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


device = 'cpu'
VOCAB_SIZE = 100
SEQ_LEN = 30
bpe_path = 'dictionary/bpe100_last.model'

bpe = yttm.BPE(model=bpe_path)
# bpe = train_bpe(dataset['query'], train_data_path="dictionary/queries_prepared.txt", vocab_size=VOCAB_SIZE, model_path=bpe_path)
model = TxtLSTM(seq_len=SEQ_LEN, num_tokens=VOCAB_SIZE, rnn_num_units=300, embedding_size=200)
model.to(device)
checkpoint = torch.load("models/lstm2_087_bpe100_nums300_emb200_batch_norm_popW_voc_last.pt",
                        map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

beam_generator = BeamGenerator(model, bpe, device=device, max_seq_len=SEQ_LEN)
greed_generator = GreedGenerator(model, bpe, device=device, max_seq_len=SEQ_LEN)


from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="template")


@app.get("/", response_class=HTMLResponse)
def get_search_page(request: Request):
    response = "fastapi.html"
    return templates.TemplateResponse(response, {"request": request})


@app.post("/search")
async def get_query_form(text: str = Form(...)):
    # result = greed_generator(seed_phrase=text, n_samples=10)
    result = beam_generator(seed_text=text, beamsize=5,
                            return_hypotheses_n=5)

    return json.dumps(result, ensure_ascii=False)
