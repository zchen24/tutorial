/*
 * Exercise Project from 
 * <<Learn C# in one day and Learn it WELL>>
 * 
 * Date: Jan 28, 2019
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;


namespace SimplePayroll
{
    public class Staff
    {
        private float hourlyRate;
        private int hWorked;

        public float TotalPay { get; protected set; }
        public float BasicPay { get; private set; }
        public string NameOfStaff { get; private set; }

        public int HoursWorked
        {
            get { return hWorked; }
            set
            {
                if (value > 0) hWorked = value;
                else hWorked = 0;
            }
        }

        public Staff(string name, float rate)
        {
            NameOfStaff = name;
            hourlyRate = rate;
        }

        public virtual void CalculatePay()
        {
            Console.WriteLine("Calculating Pay...");
            BasicPay = hWorked * hourlyRate;
            TotalPay = BasicPay;
        }

        public override string ToString()
        {
            return "Name: " + NameOfStaff + "\thourRate: " + hourlyRate +
                "\thourWorked: " + HoursWorked + "\tTotalPay: " + TotalPay;
        }
    }

    class Manager : Staff
    {
        private const float managerHourlyrate = 50f;
        public int Allowance { get; private set; }

        public Manager(string name) : base (name, managerHourlyrate) { }

        public override void CalculatePay()
        {
            //base.CalculatePay();
            base.CalculatePay();

            Allowance = 1000;
            if (HoursWorked >= 160) {
                TotalPay = BasicPay + Allowance;
            }
        }

        public override string ToString()
        {
            return "Manager: " + base.ToString();
        }
    }

    class Admin : Staff {
        private const float overtimeRate = 15.5f;
        private const float adminHourlyRate = 30.0f;
        public float Overtime { get; private set; }
       
        public Admin(string name) : base(name, adminHourlyRate) { }
        public override void CalculatePay()
        {
            base.CalculatePay();
            if (HoursWorked > 160)
            {
                Overtime = overtimeRate * (HoursWorked - 160);
            }
            else {
                Overtime = 0;
            }            
            TotalPay = BasicPay + Overtime;
        }

        public override string ToString()
        {
            return "Admin: " + base.ToString();
        }
    }

    class FileReader {
        public List<Staff> ReadFile() {
            List<Staff> staffs = new List<Staff>();
            string[] result = new string[2];
            string path = "staff.txt";
            string[] separator = { ", " };

            if (File.Exists(path)) {
                StreamReader reader = new StreamReader(path);
                while (!reader.EndOfStream) {
                    string line = reader.ReadLine();
                    Console.WriteLine(line);
                    result = line.Split(separator, StringSplitOptions.None);
                    if (result[1] == "Manager") {
                        staffs.Add(new Manager(result[0]));
                    }
                    else if (result[1] == "Admin") {
                        staffs.Add(new Admin(result[0]));
                    }
                }
                reader.Close();
            }
            else {
                Console.WriteLine("File does not exist, path = " + path);
            }
            return staffs;
        }
    }


    class PaySlip {
        private int month;
        private int year;

        enum MonthsOfYear
        {
            JAN = 1, FEB = 2, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC
        }

        public PaySlip(int payMonth, int payYear) {
            month = payMonth;
            year = payYear;
        }

        public void GeneratePaySlip(List<Staff> myStaff) {
            foreach (Staff f in myStaff) {
                // assign a value to the path variable
                string path = f.NameOfStaff + ".txt";
                bool append = false;
                using (StreamWriter writer = new StreamWriter(path, append)) {
                    writer.WriteLine("PAYSLIP FOR {0} {1}", (MonthsOfYear)month, year);
                    writer.WriteLine("==========================");
                    writer.WriteLine("Name of Staff: " + f.NameOfStaff);
                    writer.WriteLine("Hours Worked: " + f.HoursWorked);
                    writer.WriteLine("");
                    writer.WriteLine("Basic Pay: {0:C}", f.BasicPay);
                    if (f.GetType() == typeof(Manager)) { writer.WriteLine("Allowance: {0:C}", ((Manager)f).Allowance); }
                    if (f.GetType() == typeof(Admin)) { writer.WriteLine("Overtime: {0:C}", ((Admin)f).Overtime); }
                    writer.WriteLine("");
                    writer.WriteLine("==========================");
                    writer.WriteLine("Total Pay: {0:C}", f.TotalPay);
                    writer.WriteLine("==========================");
                    writer.Close();
                }
            }
        }

        public void GenerateSummary(List<Staff> myStaff) {
            // LINQ
            var staff10 =
                from f in myStaff
                where f.HoursWorked <= 10
                select f;

            string path = "summary.txt";
            bool append = false;
            using (StreamWriter writer = new StreamWriter(path, append)) {
                writer.WriteLine("Staff with less than 10 working hours");
                writer.WriteLine("");
                foreach (Staff f in staff10) {
                    writer.WriteLine("Name of Staff: {0}, Hours Worked: {1}", f.NameOfStaff, f.HoursWorked);
                }                
                writer.Close();
            }
        }

        public override string ToString()
        {
            // Customize if necessary
            return base.ToString();
        }
    }


    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello Program");

            List<Staff> myStaff;
            FileReader reader = new FileReader();
            int month = 0;
            int year = 0;
            while (year == 0) {
                Console.Write("\nPlease enter the year: ");
                try {
                    // Code to conver the input to an integer
                    year = Convert.ToInt32(Console.ReadLine());
                }
                catch (FormatException e) {
                    Console.WriteLine(e.Message);
                }
            }

            while (month == 0) {
                Console.Write("\nPlease enter the month: ");
                try
                {
                    month = Convert.ToInt32(Console.ReadLine());
                }
                catch (FormatException e) {
                    Console.WriteLine(e.Message);
                }
            }

            myStaff = reader.ReadFile();
            for (int i = 0; i < myStaff.Count; i++) {
                try
                {
                    Console.WriteLine("Enter hours worked for {0}: ", myStaff[i].NameOfStaff);
                    myStaff[i].HoursWorked = Convert.ToInt32(Console.ReadLine());
                    myStaff[i].CalculatePay();
                    Console.WriteLine(myStaff[i]);
                }
                catch (Exception e) {
                    Console.WriteLine(e.Message);
                    i--;
                }
            }

            PaySlip slip = new PaySlip(month, year);
            slip.GeneratePaySlip(myStaff);
            slip.GenerateSummary(myStaff);

            Console.ReadKey();
        }
    }
}
