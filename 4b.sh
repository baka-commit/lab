
#!/bin/bash
clear

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

# Check Java
show_prompt
show_command "java -version"
java -version || exit 1
echo

show_prompt
show_command "javac -version"
javac -version || exit 1
echo

# Check required tools
if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required."
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: unzip is required."
    exit 1
fi

# Avoid overwriting an existing project
if [ -d "studentmvc" ]; then
    echo "Error: studentmvc directory already exists."
    echo "Run this script from another directory or rename the existing project."
    exit 1
fi

# Generate Spring Boot project
show_prompt
show_command "curl -fsSL https://start.spring.io/starter.zip"

curl -fsSL "https://start.spring.io/starter.zip?type=maven-project&language=java&name=studentmvc&groupId=com.example&artifactId=studentmvc&packageName=com.example.studentmvc&packaging=jar&javaVersion=17&dependencies=web,thymeleaf" -o studentmvc.zip || exit 1

echo

# Extract project
show_prompt
show_command "unzip -q studentmvc.zip -d studentmvc"

mkdir -p studentmvc
unzip -q studentmvc.zip -d studentmvc || exit 1
rm studentmvc.zip
echo

# Enter project
show_prompt
show_command "cd studentmvc"
cd studentmvc || exit 1
echo

# Create Student.java

cat > src/main/java/com/example/studentmvc/Student.java << 'EOF'
package com.example.studentmvc;

public class Student {
    private String name = "John";
    private String rollNo = "101";
    private String department = "CSE";

    public String getName() { return name; }
    public String getRollNo() { return rollNo; }
    public String getDepartment() { return department; }
}
EOF
echo

# Create StudentController.java

cat > src/main/java/com/example/studentmvc/StudentController.java << 'EOF'
package com.example.studentmvc;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class StudentController {

    @GetMapping("/")
    public String home(Model model) {
        Student student = new Student();
        model.addAttribute("student", student);
        return "home";
    }
}
EOF
echo

# Create home.html

cat > src/main/resources/templates/home.html << 'EOF'
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Student MVC Application</title>
</head>
<body>
    <h1>Welcome to Spring MVC</h1>
    <h2>Student Information</h2>

    <p>Name: <span th:text="${student.name}"></span></p>
    <p>Roll Number: <span th:text="${student.rollNo}"></span></p>
    <p>Department: <span th:text="${student.department}"></span></p>

    <h3>Happy Learning!</h3>
</body>
</html>
EOF
echo

# Create StudentmvcApplication.java

cat > src/main/java/com/example/studentmvc/StudentmvcApplication.java << 'EOF'
package com.example.studentmvc;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class StudentmvcApplication {

    public static void main(String[] args) {
        SpringApplication.run(StudentmvcApplication.class, args);
    }
}
EOF
echo

# Create application.properties

cat > src/main/resources/application.properties << 'EOF'
spring.application.name=studentmvc
server.port=8080
EOF
echo

# Run Spring Boot application
show_prompt
show_command "chmod +x mvnw"
chmod +x mvnw

show_prompt
show_command "./mvnw spring-boot:run"
./mvnw spring-boot:run
