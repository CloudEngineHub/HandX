mkdir -p checkpoints2/
cd checkpoints2/
git lfs install
git clone https://huggingface.co/google-t5/t5-base

echo "The T5-XL will be stored in the './checkpoints' folder, named as models--google--flan-t5-xl "

cd ..