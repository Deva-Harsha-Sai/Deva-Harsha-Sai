-- SQL FINAL 
-- DDL Commands
create database HarshaExtP;

use HarshaExtP;
-- Create

create table students(
sid int primary key,
sname varchar(30) not null,
gender enum("Male","Female"),
percentage real check(percentage>50 & percentage <100),
updated enum("True", "False") default "False");

drop table students;

create table projects(
pid int primary key,
pname varchar(30) unique,
sid int not null);

-- alter

alter table projects
drop column sid;

alter table projects
add column sid int not null;

-- truncate 
truncate table projects;

-- drop
drop table projects;

-- - DML COmmands

insert into students (sid,sname,gender,percentage)values
(1,"Harsha","Male",90),
(2,"Sony","Female",69),
(3,"Baby","Female",93),
(4,"Kanna","Male",95),
(5,"Chonigadu","Female",69),
(6,"Momuu","Female",85),
(7,"Sony Nangunuri","Female",95);


-- 21 to 26 -- fav all good
-- 26 to 28 -- slow down useless things will happen -- STAY CAREFUL: 27 stay peaceful and get benifits from problems tooo  27 12:00 to 1:30 CURD
-- 28 to 30 -- peaceful af 
-- - 25 to 28 RED

insert into projects values
(1,"MLPredictions",1),
(2,"ImageProcessing",3),
(3,"AutomationBots",5),
(4,"PromptingSite",6);

-- update 

update projects
set pname = "AI Automation" where pid = 3;

-- delete
delete from projects
where pid = 3;

commit;

select * from students;

select * from projects;

select * from students where sid = any(select sid from projects where students.sid = projects.sid);
select * from students where sid = all(select sid from projects where students.sid = projects.sid);

select sname from students where sid in(1,3,5);

select * from students where exists (select sid from projects where students.sid = projects.sid);
select * from students where not exists (select sid from projects where students.sid = projects.sid);

select * from students where sname like "s%"
union
select * from students where sname like "H%";

-- aggregate functions

select gender, count(sid) from students group by gender;

select gender, avg(percentage) from students group by gender;

select sname from students order by percentage;

select count(sid) from students;

select avg(percentage) from students;

select sum(percentage) from students;

select * from students having (select sid from projects where students.sid = projects.sid);

create view couple as select sname from students where sid in(1,2);

select * from couple;

create view toppers as select sname from students where percentage > 90;

select * from toppers;

drop view couple;

delimiter @
create trigger adder
before 
insert 
on students 
for each row
set new.percentage = new.percentage+5;
@

delimiter ;

insert into students (sid,sname,gender,percentage)values
(10,"Nangunuri Deva Harsha Sai","Male",94);

select * from students;

create trigger updater
before update
on students
for each row
set new.updated = "True";

update students set percentage = 94 where sid = 1;

select * from students;

create table bin (
sid int ,
sname varchar(30),
gender enum("Male", "Female"),
percentage real);

create trigger deletor
before delete 
on students 
for each row
insert into bin (sid,sname,gender,percentage) values (old.sid,old.sname,old.gender,old.percentage);

delete from students where sid = 7;

select * from bin;

delimiter @
create procedure namefinder(a int)
begin
select sname from students where sid = a;
end @
delimiter ;

call namefinder(1);

delimiter %
create procedure cursur2(a int)
begin 
declare pomname varchar(30);
declare namefinder cursor for select sname from students where sid = a;
open namefinder;
fetch namefinder into pomname;
close namefinder;
select pomname;
end %
delimiter ;

call cursur2(2);












delimiter %
create procedure demo1(a int)
begin
declare dname varchar(30);
declare percentag real;
declare democ cursor for select sname, percentage from students where sid = a;
open democ;
fetch democ into dname, percentag;
close democ;
select dname,percentag;
end %
delimiter ;

CALL demo1(1);