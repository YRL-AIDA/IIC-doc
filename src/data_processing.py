import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os 
from PIL import Image 
import random
from tqdm import tqdm
from skimage import transform
import tifffile as tiff

alb_transforms = A.Compose(
    [
        A.Perspective(scale=(0.01, 0.03), p=0.7),  
        A.Superpixels(p_replace=(0, 0.03), p=0.7),  
        A.GaussNoise(var_limit=(0.02 * 255, 0.05 * 255), p=0.7),  
        A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.7),
    ]
)



class AlbDataset(Dataset):
    def __init__(self, labels_path, images_path, alb_transforms=None):
        self.labels_path = labels_path
        self.images_path = images_path
        self.alb_transforms = alb_transforms
        self.image_files = []
        self.labels = []
        self._scan_directory()

    def _scan_directory(self):
        labels = self.labels_path
        files = open(labels, 'r').readlines()
        print("Scanning directory...", end=" ")
        for file in files:
            row = file.split(" ") 
            img_path = row[0]
            label = row[1]

            self.labels.append(int(label))
            
            self.image_files.append(os.path.join(self.images_path, img_path))
        print("Done!")
        


    def set_transforms(self, transforms):
        self.alb_transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = tiff.imread(image_path)

        resized_image = transform.resize(image, (1000, 750))

        resized_image = resized_image.astype(np.float32)  
        if resized_image.max() > 1.0:  
            resized_image = resized_image / 255.0

        label = self.labels[idx]

        image_np = np.array(resized_image)

        if self.alb_transforms is not None:
            image_alb = self.alb_transforms(image=image_np)["image"]
            return {"original": image_np, "augmented": image_alb, "label": label}
        else:
            return {"original": image_np, "label": label}
    
def stratified_split(original_dataset, train_size, label_key="label"):
    """
    Разделяет датасет PyTorch на основе стратифицированного метода.
    Основан на функции sklearn.model_selection.train_test_split.
    Выходные наборы данных не содержат данных, но ссылаются на индексы в исходном наборе данных.

    Параметры
    ----------
    original_dataset : torch.utils.data.dataset.Dataset 
        Исходный датасет, который нужно разделить.
    train_size : float
        Доля данных, которая попадет в обучающий (первый возвращаемый) набор данных.
    label_key : str или int
        Ключ для доступа к метке из исходного набора данных.
        Если исходный набор данных возвращает словарь, это, вероятно, будет "label".
        Если исходный набор данных возвращает кортеж, то это должен быть целочисленный ключ, чаще всего равный 1.

    Возвращаемые значения
    ----------
    train_dataset : startifiedAgentDataset
        Обучающая часть разделенного набора. Размер определяется параметром train_size.
    test_dataset : startifiedAgentDataset
        Тестовая часть разделенного набора. Размер равен len(original_dataset) - train_size.
    """


    labels = [x[label_key] for x in original_dataset]
    indices = np.arange(len(labels))
    indices_and_labels = np.vstack([indices, labels]).transpose()
    train_indices_and_labels, test_indices_and_labels = train_test_split(
        indices_and_labels, train_size=train_size, stratify=labels
    )
    train_indices = train_indices_and_labels[:, 0].astype(int)
    test_indices = test_indices_and_labels[:, 0].astype(int)
    train_dataset = startifiedAgentDataset(original_dataset, train_indices)
    test_dataset = startifiedAgentDataset(original_dataset, test_indices)

    return train_dataset, test_dataset

class startifiedAgentDataset(Dataset):
    """
    Набор данных, создаваемый функцией stratified_split.
    Этот набор данных получает данные непосредственно из исходного экземпляра набора данных.
    Он может обращаться только к определенной части данных исходного набора данных.

    Атрибуты
    ----------
    original_dataset : torch.utils.data.dataset.Dataset или подобный
        Исходный набор данных, из которого берутся данные.
    indices_list : список
        Список индексов, которые могут быть доступны из исходного набора данных.
    """


    def __init__(self, original_dataset, indices_list):
        super().__init__()
        self.original_dataset = original_dataset
        self.indices_list = indices_list

    def __len__(self):
        return len(self.indices_list)

    def __getitem__(self, idx):
        original_idx = self.indices_list[idx]
        return self.original_dataset[original_idx]
    
def create_dataset_arrays(
    alb_transforms=None, aug_number=1, labels_path='.', images_path='.', batch_size=256, num_workers=1
):


    dataset_alb = AlbDataset(labels_path=labels_path, images_path=images_path)
    dataloader = DataLoader(
        dataset_alb, batch_size=len(dataset_alb), shuffle=False, num_workers=num_workers
    )
    print(np.array(dataloader).shape)

    print("Fetching original dataset...", end=" ")
    originals_array = np.array(next(iter(dataloader))["original"])
    labels_array = np.array(next(iter(dataloader))["label"])
    print("Done!")

    dataset_alb.set_transforms(alb_transforms)
    aug_arrays = []
    dataloader = DataLoader(
        dataset_alb, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    for aug_idx in range(aug_number):
        print("Making aug #%i" % aug_idx)
        aug_arrays.append(compose_array_from_dataloader(dataloader, key="augmented"))

    return originals_array, labels_array, aug_arrays

class RAMAug(Dataset):
    """
    Набор данных, содержащий исходные и аугментированные данные в оперативной памяти.

    Атрибуты
    ----------
    original_dataset : numpy.ndarray
        Форма (N,...), где N — количество образцов данных, а "..." обозначает форму одного образца данных. Массив с исходными изображениями.
    labels : numpy.ndarray
        Форма (N,). Массив с метками классов.
    aug_datasets : список numpy.ndarray
        Список аугментированных  наборов данных, имеющих ту же форму, что и original_dataset.
    aug_number : int
        Количество аугментаций.
    """


    def __init__(
        self,
        augs_files=None,
        alb_transforms=A.Compose([]),
        aug_number=1,
        labels_path=".",
        images_path=".",
        aug_batch_size=1024,
        aug_num_workers=1,
    ):
        """
        Инициализация набора данных

        Параметры
        ----------
        augs_files : список строк
            Список путей к бинарным файлам np (numpy) с аугментациями. Если этот параметр указан,
            другие параметры можно не передавать.
        alb_transforms : albumentations.core.composition.Compose
            Композиция преобразований Albumentations. Игнорируется, если указан параметр augs_files.
        aug_number : int
            Количество создаваемых аугментаций. Игнорируется, если указан параметр augs_files.
        target_dir : str
            Директория для хранения набора данных. Игнорируется, если указан параметр augs_files.
        batch_size : int
            Размер батча, используемого при аугментации. Игнорируется, если указан параметр augs_files.
        num_workers : int
            Количество потоков CPU для использования при аугментации. Игнорируется, если указан параметр augs_files.
        """


        if augs_files is not None:
            aug_number = 0

        self.original_dataset, self.labels, self.aug_datasets = create_dataset_arrays(
            alb_transforms=alb_transforms,
            aug_number=aug_number,
            labels_path=labels_path,
            images_path=images_path,
            batch_size=aug_batch_size,
            num_workers=aug_num_workers,
        )

        self.aug_number = aug_number

        if augs_files is not None:
            print("Loading augmented datasets...", end=" ")
            self.aug_datasets = [np.load(x) for x in augs_files]
            self.aug_number = len(augs_files)
            print("Done!")

    def save_augs(self, dataset_dir):
        

        print("Saving augs in %s..." % dataset_dir, end=" ")
        for aug_idx, aug_array in enumerate(self.aug_datasets):
            file_path = os.path.join(dataset_dir, "data_aug_" + str(aug_idx) + ".np")
            with open(file_path, "wb") as file:
                np.save(file, aug_array)
        print("Done!")

    def __len__(self):
        return self.original_dataset.shape[0]

    def __getitem__(self, idx):
        """
        Возвращает образец набора данных и индекс `idx`.

        Возвращаемые значения
        -------
        return_dict : dict
            Словарь с следующими ключами:
            "original" : numpy.ndarray
                Исходное изображение, взятое из self.original_dataset.
            "aug" : numpy.ndarray
                Аугментированное изображение, случайно выбранное из одного из self.aug_datasets.
            "label" : int
                Метка (label) изображения, взятая из self.labels.
        """

        type(idx)

        original_image = np.array(
            self.original_dataset[idx: idx + 1, :, :], copy=True, dtype=np.float16
        )
        #print("or_img: ", original_image.transpose(0,3,1,2).shape)

        label = np.array(self.labels[idx], copy=True, dtype=np.float16)

        return_dict = {"original": original_image, "label": label}

        if self.aug_number > 0:
            aug_version = int(self.aug_number * random.random())
            aug_image = np.array(
                self.aug_datasets[aug_version][idx: idx+1, :, :],
                copy=True,
                dtype=np.float16,
            )
            #print("aug_img: ", aug_image.transpose(0,3,1,2).squeeze().shape)
            return_dict["aug"] = aug_image

        return return_dict
    

def compose_array_from_dataloader(dataloader, key="original"):
    """
    Создает массив numpy из даталоадера PyTorch.

    Параметры
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        Исходный даталоадер, предоставляющий данные, которые будут преобразованы в массив numpy.
    key : str
        Определяет, что будет преобразовано.
        "original" — исходный набор данных,
        "augmented" — аугментированные (измененные) изображения,
        "label" — аннотации (метки).

    Возвращает
    -------
    output_array : numpy.array
        Выходной массив.
        Если параметр "key" равен "original" или "augmented", форма массива будет (N, H, W), где N = длина даталоадера, H и W — ширина и высота изображений.
        Если параметр "key" равен "label", форма выходного массива будет (N,), где N = длина даталоадера.
    """


    sample = dataloader.dataset[0][key]

    if key == "label":
        dtype = int
        output_shape = [len(dataloader.dataset)]
    else:
        dtype = np.float32
        output_shape = [len(dataloader.dataset)] + list(sample.shape)

    output_array = np.zeros(output_shape, dtype=dtype)
    output_array.setflags(write=True)
    global_batch_size = dataloader.batch_size

    with tqdm(total=len(dataloader)) as pbar:
        for idx, batch in enumerate(dataloader):
            array_to_add = batch[key].numpy()
            batch_size = array_to_add.shape[0]
            output_array[
                global_batch_size * idx : global_batch_size * idx + batch_size
            ] = array_to_add
            pbar.update(1)

    return output_array