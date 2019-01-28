using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    // enum
    enum DaysOfWeek
    {
        Sun, Mon, Tues, Wed, Thurs, Fri, Sat
    }

    class Customer
    {
        public string name;
        public decimal balance;

        public Customer() {
        }
    }

    class BaseMember {
        protected int annualFee;
        private string name;
        private int memberID;
        private int memberSince;

        public override string ToString()
        {
            return "\nName: " + name +
                "\nMember ID: " + memberID +
                "\nMember Since: " + memberSince +
                "\nTotal Annual Fee: " + annualFee;
        }

        public BaseMember() {
            Console.WriteLine("Base Constructor with no parameter");
        }

        public BaseMember(string pName, int pMemberID, int pMemberSince) {
            Console.WriteLine("Base Constructor with 3 parameters");
            name = pName;
            memberID = pMemberID;
            memberSince = pMemberSince;
        }

        public virtual void Play() {
            Console.WriteLine("BaseMember Play()");
        }

        public void FuncOverload() {
            Console.WriteLine("FuncOverload: parameterless");
        }

        public void FuncOverload(string str) {
            Console.WriteLine("FuncOverload: string parame = " + str);
        }
    }

    class NormalMember : BaseMember
    {
        public NormalMember()
        {
            Console.WriteLine("Derived constructor with no parameter");
        }

        public override void Play() {
            Console.WriteLine("Calling: base.Play()");
            base.Play();
            Console.WriteLine("NormalMember Play()");
        }
    }





    class Program
    {
        static void Main(string[] args)
        {
            // ----------------------
            // Hello World
            // ----------------------
            Console.WriteLine("Hello Worlld!");

            // ----------------------
            // Variables
            // ----------------------
            byte myByte = 20;
            int myInt = 510;

            float myFloat = 10.0f;
            double myDouble = 10.0;
            decimal myDecimal = 10.0m;

            char myChar = 'B';
            bool myBool = true;

            int counter = 2;
            Console.WriteLine("counter   = " + counter);
            Console.WriteLine("counter++ = " + counter++);
            Console.WriteLine("counter   = " + counter);
            Console.WriteLine("++counter = " + ++counter);

            counter = (int)10.5;
            Console.WriteLine("count = (int)10.5 = " + counter);

            // ----------------------
            // Arrays
            // ----------------------
            int[] a0 = { 5, 4, 3, 2, 1 };
            int[] a1 = { 1, 2, 3, 4, 5 };
            Console.WriteLine("a0 = {0}", string.Join(", ", a0));
            Console.WriteLine("a1 = {0}", string.Join(", ", a1));

            Array.Copy(a0, a1, 2);
            Console.WriteLine("a0.Length = " + a0.Length);
            Console.WriteLine("a1 = {0}", string.Join(", ", a1));

            Array.Sort(a0);
            Console.WriteLine("Array.Sort(a0) = {0}", string.Join(", ", a0));

            int index4 = Array.IndexOf(a1, 4);
            Console.WriteLine("Array.IndexOf(a1, 4) = {0}", index4);
            Console.WriteLine("Array.IndexOf(a1, 100) = {0}", Array.IndexOf(a1, 100));

            char[] message = { 'H', 'e', 'l', 'l', 'o', '\n' };

            // ----------------------
            // String
            // ----------------------
            Console.WriteLine("\n\n=====  string  =====");
            string str1 = "Hello String1";
            Console.WriteLine(str1);
            str1 = str1 + " + a name";
            Console.WriteLine(str1);
            Console.WriteLine("str1.Length = {0}", str1.Length);
            Console.WriteLine("str1.Substring(0, 5) = {0}", str1.Substring(0, 5));
            Console.WriteLine("str1.Contains(\"Hello\") = {0}", str1.Contains("Hello"));
            Console.WriteLine("str1.Equals(\"Hello\") = {0}", str1.Equals("Hello"));
            string[] str1Splits = str1.Split(' ');
            Console.Write("str1.Split = ");
            for (int i = 0; i < str1Splits.Length; i++) {
                Console.Write(str1Splits[i] + " || ");
            }
            Console.WriteLine("");


            // convert string to numbers
            Int32 int45 = Convert.ToInt32("45");
            Console.WriteLine("Convert.ToInt32(\"45\") = {0}", int45);


            // ----------------------
            // Flowcontrol
            // ----------------------
            Console.WriteLine("\n\n=====  flow control  =====");
            // if else 
            if (true == true) {
                Console.WriteLine("ture is always equal to true");
            } else {
                Console.WriteLine("Hence this line will never be printed");
            }

            // switch 
            int mySwitchVariable = 10;
            switch (mySwitchVariable) {
                case 1:
                    Console.WriteLine("case = 1");
                    break;
                default:
                    Console.WriteLine("case default");
                    break;
            }

            // for loop
            Console.WriteLine("\n\n===== for loop =====");
            for (int i = 0; i < message.Length; i++) {
                Console.Write(message[i]);
            }

            // foreach loop
            Console.WriteLine("===== foreach  =====");
            foreach (char c in message) {
                Console.Write(c);
            }

            // while 
            Console.WriteLine("\nStart while loop\n");
            int whileCounter = 5;
            while (whileCounter > 2) {
                Console.Write(whileCounter + " ");
                whileCounter--;
            }
            Console.WriteLine("\nEnd while loop\n");

            // do while 
            whileCounter = 10;
            do {
                Console.Write(whileCounter + " ");
                whileCounter--;
            } while (whileCounter > 5);
            Console.WriteLine("\nEnd do while\n");

            // break/continue 
            for (int i = 0; i < 5; i++) {
                if (i > 3) {
                    break;
                } else {
                    Console.Write(i + " ");
                }
            }
            Console.WriteLine("\nEnd break\n");

            for (int i = 0; i < 5; i++) {
                if (i == 2) {
                    continue;
                } else {
                    Console.Write(i + " ");
                }
            }
            Console.WriteLine("\nEnd continue\n");

            // try catch finally
            try {
                Convert.ToInt32("hello");
            }
            catch (Exception e) {
                Console.WriteLine(e.Message);
            }
            finally {
                Console.WriteLine("finally: End of try-catch-finally");
            }


            // ----------------------
            // Enum & Struct
            // ----------------------
            DaysOfWeek myDays = DaysOfWeek.Sat;
            Console.WriteLine(myDays);
            Console.WriteLine((int)myDays);


            // ----------------------
            // LINQ
            // ----------------------
            int[] numbers = { 0, 1, 2, 3, 4 };
            var evenNumbers =
                from num in numbers
                where (num % 2) == 0
                select num;

            Console.Write("Even numbers =");
            foreach (int n in evenNumbers) {
                Console.Write(" {0}", n);
            }
            Console.WriteLine(";");

            // ----------------------
            // FILE IO
            // ----------------------
            Console.WriteLine("===== file I/O =====");

            string fileName = "file.txt";
            bool append = false;
            // context management, similar to with statement
            using (StreamWriter writer = new StreamWriter(fileName, append)) {
                writer.WriteLine("1 2 3");
                writer.WriteLine("4 5 6");
                writer.Close();
            }

            if (File.Exists(fileName)) {
                StreamReader reader = new StreamReader(fileName);
                while (!reader.EndOfStream)
                {
                    Console.WriteLine(reader.ReadLine());
                }
                reader.Close();
            }
            else {
                Console.WriteLine("File {} does not exist", fileName);
            }

            // ----------------------
            // OOP
            // ----------------------
            Console.WriteLine("\n===============");
            Console.WriteLine(" OOP ");
            Console.WriteLine("===============");
            NormalMember nMember = new NormalMember();
            Console.WriteLine(nMember);

            // function override
            Console.WriteLine("\nFunction overriding");
            nMember.Play();

            // function overload
            Console.WriteLine("\nFunction overloading");
            nMember.FuncOverload();
            nMember.FuncOverload("A parameter");

            // type check
            Console.WriteLine("\nClass type check");
            if (nMember.GetType() == typeof(BaseMember)) {
                Console.WriteLine("nMember == BaseMember: True");
            }
            else {
                Console.WriteLine("nMember == BaseMember: False");
            }
            if (nMember.GetType() == typeof(NormalMember)) {
                Console.WriteLine("nMember == NormalMember: True");
            }
            else {
                Console.WriteLine("nMember == NormalMember: False");
            }

            // Keep the console window open in debug mode 
            Console.WriteLine("Press any key to exit");
            Console.ReadKey();
        }
    }
}
