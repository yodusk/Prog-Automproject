# Implementation of experiment published in paper ["Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs"](https://www.aclweb.org/anthology/P19-1466.pdf)

## Постановка задачи
Граф знаний предстваляется G = (E, R), где E и R предствавляют множество сущностей(узлов)
и отношений (ребер). Тройка  (es, r, eo) представляется
в виде ребра r между es и er.

Эмбеддинговые модели обучаются эффективному представлению сущностей, так что функция f от входящей тройки t = (es, r, eo), возвращает
вероятность t - правильная тройка.

Архитектура сети представлена encoder-decoder моделью, где GAT выступает
в роди encoder, а сверточная нейронная сеть декодером.

## Датасеты
#### Freebase: FB15k-237
- Entities: 14,541
- Relations: 237
- Training triples: 272,115
- Validation triples: 17,535
- Test triples: 20,466

Оригинальный датасет страдал от утечки тестовых данных из-за двухсторонних связей. В работе используется модифицированная версия
датасета, предстваленная Toutanova et al, где такие связи были убраны.

#### Wordnet: WN18RR
- Entities: 40,943
- Relations: 11
- Training triples: 86,835
- Validation triples: 3,034
- Test triples: 3,134

Все датасеты находятся в папке Data.


### Зависимости
- numpy
- pandas
- torch
- scikit-learn
- scipy

Так же они указаны в requirements.txt, и устанавливаются при запуске mldev.

Код для обучения и валидации находится в файле main.py.

Автоматического тестирования не предусмотрена, результаты эксперимента
выводятся в стандартный вывод.

## Постановка эксперимента
**Hypothesis**
Улучшение относительно state-of-the-art решения на датасете FreeBase.

Два датасета: Freebase: FB15k-237 and Wordnet: WN18RR

### Описание параметров
* —**data**: Путь к директории с данными
* —**epochs_gat**: Кол-во эпох для обучения GAT модели
* —**epochs_conv**: Кол-во эпох для обучения Conv модели.
* —**lr**: Исходный learning rate.
* —**weight_decay_gat**: L2 регуляризация для gat.
* —**weight_decay_conv**: L2 регуляризация для conv.
* —**get_2hop**: Использовать pickle файл содержащий 2 hop соседей.
* —**use_2hop**: Использовать 2-hop соседей при обучении
* —**partial_2hop**: Использовать только 1 2-hop соседа на один узел в процессе обучения
* —**output_folder**: Путь к директории для сохранения моделей
* —**batch_size_gat**: Размер батча для GAT модели
* —**valid_invalid_ratio_gat**: Отношение действительных и недействительных троек для GAT модели
* —**drop_gat**: Вероятность дропаута для GAT
* —**alpha**: Alpha для leakyRelu в GAT
* —**margin**: margin для функции потерь Хинджа
* —**batch_size_conv**: Размер батча для Conv модели
* —**alpha_conv**: Alpha для leakyRelu в Conv
* —**valid_invalid_ratio_conv**: Отношение действительных и недействительных троек для Conv модели
* —**out_channels**: Кол-во выходных каналов для Conv
* —**drop_conv**: Вероятность дропаута для Conv


### FreeBase параметры
* epochs_gat: 3000
* epochs_conv: 200
* weight_decay_gat: 0.00001
* get_2hop: True
* partial_2hop: True
* batch_size_gat: 272115
* margin: 1
* out_channels: 50
* drop_conv: 0.3
* weight_decay_conv: 0.000001

### WordNet параметры
* epochs_gat: 3600
* epochs_conv: 200
* weight_decay_gat: 0.00001
* get_2hop: True
* partial_2hop: False
* batch_size_gat: 86835
* margin: 5
* out_channels: 500
* drop_conv: 0.0
* weight_decay_conv: 0.000001


## Протокол обучения
Создаются два множества недействительных троек, каждый раз заменяя 
либо головную либо хвостовую сущность недействительной. Случайно выбирается одинаковое
количество недействительных троек из обоих множеств. Эмбеддинги для 
сущностей и отнощений генерируются TransE для инциализации эмбеддингов модели.

## Протокол оценки

В задаче предсказания связей стоит задача предсказать тройку
(ei, rk, ej), когда ei или ej отсутствует.

Генерируется множество недействительных троек размером (N − 1) для каждой сущности, заменяя ее 
каждой другой сущностью, такой что ei0 ∈ E \ ei. Далее каждой такой тройке присваивается оценка и сортируется по убыванию.
Таким образом получается ранк действительной тройки (ei, rk, ej ).

В процессе оценки тройки присутствующие в обучающей, тестовой и валидационной выборке исключаются.

Измеряются следующие метрики.
-**Mean Reciprocal Rank (Среднеобратный ранг):**
-**mean rank (MR)**
-**Hits at k (H@k):**
Отношение действительных сущностей к недействительным, которые есть в топ K записях.

## Воспроизведение эксперимента и результаты.
### Воспроизведение
- Склонировать данный репозиторий
- Установить Mldev из данного [репозитория](https://gitlab.com/mlrep/mldev)
- `cd <project folder>`
- `mldev init -r .`
- `mldev run -f experiment.yml <pipeline name>` 
#### Pipelines
- FB15K_RUN
- WordNet_Run
- Kinship_run
- Nell_run

Результаты печатаются в стандартный вывол.

#### Воспроизведение с помощью Google Colab

- Откройте .ipynb тетрадку из директории Notebook в Google Colab
- Запустите первую ячейку для установки MLdev
- Далее есть четыре сценария выполнения эксперимента на различных
датасетах.

**Default network params are:**
- data: ./data/WN18RR/
- epochs_gat: 3600
- epochs_conv: 200
- weight_decay_gat: 5e-6
- weight_decay_conv: 1e-5
- pretrained_emb: True
- embedding_size: 50
- lr: 1e-3
- get_2hop: False
- use_2hop: True
- partial_2hop: False
- output_folder: ./checkpoints/wn/out/

**Default GAT params are:**
- batch_size_gat: 86835
- valid_invalid_ratio_gat: 2
- drop_GAT: 0.3
- alpha: 0.2
- entity_out_dim: [100, 200]
- nheads_GAT: [2, 2]
- margin: 5

**Default Conv network params are:**
- batch_size_conv: 128
- alpha_conv: 0.2
- valid_invalid_ratio_conv: 40
- out_channels: 500
- drop_conv: 0.0

_Параметры могут быть изменены в experiment.yml_


### Результаты
| Dataset/Results  | MR | MRR   | @10 hits | @3 hits |  @1 hits |
| ---------------- | --- | ---- | -------- |  ------ | -------- |
| Предоставленные авторами (WordNet) | 1940  | 0.440  | 0.581  | 0.483  | 0.361  |
| Полученные (WordNet)  | 2336.783 | 0.331 | 0.557  | 0.44  | 0.19 |
| Предоставленные авторами (FreeBase) | 210  | 0.518  | 0.626  | 0.54  | 0.46  |
| Полученные (FreeBase) | 233.36   | 0.35 | 0.557  | 0.412 | 0.237 |

Эксперимент конфигурируется в файле experiment.yml
Эксперимент проводился с использованием Colab Pro
- **GPU** T4 & P100
- **Memory** 25 GB

# Hypothesis result
Результаты выполнения на датасете FreeBase превысили лучший результат, достигнутый 
ConvE (Dettmers et al., 2018) в @1 hits 22.5%

При использовании бесплатной версии Colab Pro не хватило памяти при проведении эксперимента с
использованием датасета WordNet.

