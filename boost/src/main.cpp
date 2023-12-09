#include <boost/regex.hpp>
#include <iostream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

int main()
{
    std::string line = "Subject: Re: 1232121312";
    boost::regex pat( "^Subject: (Re: |Aw: )*(.*)" );

    {
        //std::getline(std::cin, line);
        boost::smatch matches;
        if (boost::regex_match(line, matches, pat))
            std::cout << matches[2] << std::endl;
    }

    boost::filesystem::path p(".");

    if(boost::filesystem::is_directory(p)) {
        std::cout << p << " is a directory containing:\n";

        for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {}))
            std::cout << entry << "\n";
    }

}