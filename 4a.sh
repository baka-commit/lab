#!/bin/bash

GREEN='\033[1;32m'
PURPLE='\033[1;35m'
YELLOW='\033[1;33m'
RESET='\033[0m'

show_prompt() {
    printf "${GREEN}%s@%s${RESET} ${PURPLE}UCRT64${RESET} ${YELLOW}%s${RESET}\n" \
        "$(whoami)" "$(hostname)" "$(pwd)"
}

show_command() {
    printf '$ %s\n' "$1"
}


cat > Main.java << 'EOF'
class Student {
    int id;
    String name;
    int age;

    void displayDetails() {
        System.out.println("Student ID: " + id);
        System.out.println("Student Name: " + name);
        System.out.println("Student Age: " + age);
    }

    void study() {
        System.out.println(name + " is studying.");
    }
}

public class Main {
    public static void main(String[] args) {
        Student s1 = new Student();

        s1.id = 101;
        s1.name = "Praveen";
        s1.age = 21;

        s1.displayDetails();
        s1.study();
    }
}
EOF

echo

# Compile Java program
show_prompt
show_command "javac Main.java"
javac Main.java || exit 1
echo

# Run Java program
show_prompt
show_command "java Main"
java Main