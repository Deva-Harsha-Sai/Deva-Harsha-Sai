#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>

int main() {
    int file = 0;
    if ((file = open("data.txt", O_RDONLY)) < 0)
        return 1;

    char buffer[20];
    ssize_t bytesRead = read(file, buffer, sizeof(buffer) - 1);
    if (bytesRead < 0)
        return 1;
    buffer[bytesRead] = '\0';
    printf("%s\n", buffer);

    if (lseek(file, 5, SEEK_SET) < 0)
        return 1;

    bytesRead = read(file, buffer, sizeof(buffer) - 1);
    if (bytesRead < 0)
        return 1;
    buffer[bytesRead] = '\0';
    printf("%s\n", buffer);

    close(file);

    return 0;
}

