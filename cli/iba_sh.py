import os
import glob

FOLDER_PATH = "/home/henry/learning2/LIRA-Federated-Learning/experiments/fl-defenses"

files_timg = glob.glob(os.path.join(FOLDER_PATH, "**", "timage*"), recursive=True)

print(files_timg)
print(len(files_timg))
for idf, ft in enumerate(files_timg):
    print(idf, os.path.basename(ft))

def update_the_change(test_sh_file='test.sh', new_file_write='new_test.sh', type_exp="iba"):
    # Open the input file for reading
    with open(test_sh_file, 'r') as input_file:
        lines = input_file.readlines()

    # Replace the desired line
    # updated_lines = [line.replace("--project_frequency 1", "--project_frequency 2") for line in lines]
    # updated_lines = [line.replace("--project_frequency 1", "--project_frequency 2") for line in lines]
    
    updated_lines = [line.replace("--attack_method blackbox", "--attack_method pgd") for line in lines]
    # ["iba", "iba_pgd", "iba_pgd_mr"]
    
    instance_wandb = f"{os.path.basename(new_file_write)[:-3]}"
    # print(instance_wandb)
    # return
    up_lines = []
    for line in updated_lines:
        if "model_replacement" in line:
            if type_exp == "iba":
                up_lines.append("--model_replacement False \\\n")
            else:
                up_lines.append("--model_replacement True \\\n")
                
        elif "project_frequency" in line:
            up_lines.append("--project_frequency 2 \\\n")
        elif "attack_freq" in line:
            up_lines.append("--attack_freq 10 \\\n")
        elif "--lr" in line:
            up_lines.append("--lr 0.001 \\\n")
        elif "--batch-size" in line:
            up_lines.append("--batch-size 256 \\\n")
        elif "--test-batch-size" in line:
            up_lines.append("--test-batch-size 256 \\\n")

        # --batch-size 256 \
        # --test-batch-size 256 \
               
        elif "attack_method" in line:
            if type_exp == "iba":
                up_lines.append("--attack_method blackbox \\\n")
            else:
                up_lines.append("--attack_method pgd \\\n")
        elif "instance" in line:
            up_lines.append(f"--instance defense__18_May__{instance_wandb} \\\n")        
        else:
            up_lines.append(line)
        
    # -model_replacement True

    # Write the updated content to a new file
    os.makedirs(os.path.dirname(new_file_write), exist_ok=True)
    with open(new_file_write, 'w') as output_file:
        output_file.writelines(up_lines)
    print(f"Done write to files: {new_file_write}")
    

def generate_file_sh_iba_lv1(files_timg):
    type_exps = ["iba", "iba_pgd", "iba_pgd_mr"]
    for ty in type_exps:
        print(f"Process with : {ty}")
        for idf, ft in enumerate(files_timg):    
            new_ft = ft.replace("fl-defenses", "timagenet-fl-defenses")
            new_ft = new_ft.replace(".sh", f'__{ty}.sh')
            
            
            update_the_change(ft, new_ft, ty)
            # update_the_change(ft, new_ft, "iba_pgd")
            # update_the_change(ft, new_ft, "iba_pgd_mr")
        
             
    # c2: -attack_method blackbox \ = pgd

    # c3: -attack_method blackbox \ = pgd

    # -model_replacement True \

    # -attack_freq 10 \
    pass

generate_file_sh_iba_lv1(files_timg)