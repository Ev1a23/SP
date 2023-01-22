#include <stdio.h>
typedef struct node {
int val;
struct node *left;
struct node *right;
} Node;

typedef struct student {
char grade;
char* last_name;

} Student;

typedef struct _list{
    int value;
    struct _list* next;
}list;

int free_tree(Node* root)
{   
    if(root == NULL)
    {
        return 0;
    }
    static int counter = 0;
    counter++;
    if(root->left == NULL && root->right == NULL)
    {
        free(root);
    }
    else
    {
        if(root->left != NULL)
        {
            free_tree(root->left);
        }
        if(root->right != NULL)
        {
            free_tree(root->right);
        }
    }
    return counter;
}


void shiftleft(unsigned int* arr)
{
    unsigned int mask;
    mask = arr[0] >> (sizeof(unsigned int)*8 -3);
    printf("%d",mask);
    arr[0] = arr[0] <<3;
    arr[1] = arr[1] <<3;
    arr[1] = arr[1] ^ mask;
}
list* total(list* head)
{
    list* prev = NULL;
    list* new_head = NULL;
    while(head!=NULL)
    {
        list* new_item = (list*) malloc(sizeof(list));
        new_item->value = head->value;
        if(prev == NULL)
        {
            new_head = new_item;
        }
        else
        {
            new_item->value += prev->value;
            prev->next = new_item;
        }
        
        prev = new_item;
        head = head->next;
    }
    prev-> next = NULL;
    return new_head;
}

int main(int argc, char** argv)
{
    Node l = {1,NULL,NULL};
    Node root = {2,&l, NULL};
    // printf("%d",free_tree(&root));


    // unsigned int a[] = {13u, 30u};
    // shiftleft(a);
    // printf("%u,%u", a[0],a[1]);
    // int j = 3;
    // const int *p = &j;
    // int **n = &p;
    // **n =5;
    // printf("%d",j);
    list x4 = {4,NULL};
    list x3 = {3, &x4};
    list x2 = {2, &x3};
    list x1 = {1, &x2};
    list* new = total(&x1);
    list h[5] = {{4,NULL}};
    list* zevel = (list*) calloc(1,sizeof(list));
    if(zevel->next == NULL)
    {
        printf("nevela\n");
    }

    while(new !=NULL)
    {
        printf("%d\n",new->value);
        new = new->next;
    }
   printf("%d",(&h[1])->next);


}