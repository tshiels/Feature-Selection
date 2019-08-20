#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <time.h>

using namespace std;

double e_distance(vector<double> x, vector<double> y, vector<int> curr_set, int j,
    bool fwd);
    
double loo_nn(vector< vector<double> > data, vector<int> curr_set, int j, 
    bool fwd);
    
void forward_selection(vector< vector<double> > data);
void backward_elimination(vector< vector<double> > data);

void custom_alg(vector< vector<double> > data)
{
    vector<int> top_three(3);
    vector<double> accuracies;
    vector<int> current_set_of_features;
    current_set_of_features.resize(1);
    double max_acc = 0;
    
    //test all features individually
    for (int i = 1; i < data.at(0).size(); ++i)
    {
        current_set_of_features.at(0) = i;
        double temp_acc = loo_nn(data, current_set_of_features, 0, false);
        accuracies.push_back(temp_acc);
    }
    
    //capture top three features
    for (int i = 0; i < accuracies.size(); ++i)
    {
        if (accuracies.at(i) > max_acc)
        {
            max_acc = accuracies.at(i);
            top_three.at(0) = i + 1;
        }
    }
    max_acc = accuracies.at(0);
    for (int i = 0; i < accuracies.size(); ++i)
    {
        if ((accuracies.at(i) > max_acc) && (i + 1 != top_three.at(0)))
        {
            max_acc = accuracies.at(i);
            top_three.at(1) = i + 1;
        }
    }
    max_acc = accuracies.at(0);
    for (int i = 0; i < accuracies.size(); ++i)
    {
        if ((accuracies.at(i) > max_acc) && (i + 1 != top_three.at(1)) &&
            (i + 1 != top_three.at(0)))
        {
            max_acc = accuracies.at(i);
            top_three.at(2) = i + 1;
        }
    }
    
    //create new data matrix with only top three features
    vector< vector<double> > data2(data.size());
    vector<double> temp_data(4);
    for (int i = 0; i < data.size(); ++i)
    {
        temp_data.at(0) = data.at(i).at(0);
        temp_data.at(1) = data.at(i).at(top_three.at(0));
        temp_data.at(2) = data.at(i).at(top_three.at(1));
        temp_data.at(3) = data.at(i).at(top_three.at(2));
        
        data2.at(i) = temp_data;
    }
    
    forward_selection(data2);
    return;
}

//check to see if two features match
bool intersect(vector<int> curr_feats, int index)
{
    for (int i = 0; i < curr_feats.size(); ++i)
    {
        if (curr_feats.at(i) == index)
        {
            return true;
        }
    }
    return false;
}

//used to remove a feature during backward elimination
vector<int> remove_feat(vector<int> feat, int index)
{
    for (int i = 0; i < feat.size(); ++i)
    {
        if (feat.at(i) == index)
        {
            feat.erase(feat.begin() + i);
            return feat;
        }
    }
    return feat; 
}

void backward_elimination(vector< vector<double> > data)
{
    vector<int> current_set_of_features;
    vector<int> best_feats;
    double acc_total = 0;
    
    //fill set with all features
    for (int i = 1; i < data.at(0).size(); ++i)
    {
        current_set_of_features.push_back(i);
    }
    
    //l.o.o nearest neighbor starting with n - 1 features, and
    //check best accuracy
    for (int i = 1; i < data.at(0).size() - 1; ++i)
    {
        int feat_to_remove;
        double best_acc_so_far = 0;
        
        for (int j = 1; j < data.at(0).size(); ++j)
        {
            if (intersect(current_set_of_features, j))
            {
                vector<int> temp = remove_feat(current_set_of_features, j);
                cout << "Using feature(s) {";
                for (int k = 0; k < temp.size() - 1; ++k)
                {
                    cout << temp.at(k) << ",";
                }
                cout << temp.at(temp.size() - 1) << "} accuracy is ";
                
                double acc = loo_nn(data, temp, j, false);
                cout << acc << endl;
                if (acc > best_acc_so_far)
                {
                    best_acc_so_far = acc;
                    feat_to_remove = j;
                }
            }
        }
        current_set_of_features = remove_feat(current_set_of_features, 
            feat_to_remove);
        
        if (best_acc_so_far > acc_total)
        {
            acc_total = best_acc_so_far;
            best_feats = current_set_of_features;
        }
        else
        {
            cout << "(Warning, accuracy has decreased! Continuing search in ";
            cout << "case of local maxima)" << endl;
        }
        cout << "Feature set {";
        for (int k = 0; k < current_set_of_features.size() - 1; ++k)
        {
            cout << current_set_of_features.at(k) << ",";
        }
        cout << current_set_of_features.at(current_set_of_features.size() - 1);
        cout << "} was best, accuracy is ";
        cout << best_acc_so_far * 100 << "%" << endl;
        cout << endl;
    }
    
    cout << "Finished search! The best feature subset is {";
    for (int k = 0; k < best_feats.size() - 1; ++k)
    {
        cout << best_feats.at(k) << ",";
    }
    cout << best_feats.at(best_feats.size() - 1);
    cout << "} which has an accuracy of " << acc_total * 100 << "%" << endl;
    
    return;
}

void forward_selection(vector< vector<double> > data)
{
    vector<int> current_set_of_features;
    vector<int> best_feats; 
    double acc_total = 0;
    
    //for every feature
    for (int i = 1; i < data.at(0).size(); ++i)
    {
        cout << "On level " << i << " of the search tree" << endl;
        int feat_to_add_this_level = 0;
        double best_acc_so_far = 0;
        
        //l.o.o nearest neighbor 
        for (int j = 1; j < data.at(0).size(); ++j)
        {
            //do not compare a feature to itself
            if (!intersect(current_set_of_features, j))
            {
                cout << "Using feature(s) {";
                for (int k = 0; k < current_set_of_features.size(); ++k)
                {
                    cout << current_set_of_features.at(k) << ",";
                }
                cout << j << "} accuracy is ";
                
                double acc = loo_nn(data, current_set_of_features, j, true);
                
                cout << acc * 100 << "%" << endl;
                if (acc > best_acc_so_far)
                {
                    best_acc_so_far = acc;
                    feat_to_add_this_level = j;
                }
            }
        }
        //check total accuracy
        current_set_of_features.push_back(feat_to_add_this_level);
        if (best_acc_so_far > acc_total)
        {
            acc_total = best_acc_so_far;
            best_feats = current_set_of_features;
        }
        else
        {
            cout << "(Warning, accuracy has decreased! Continuing search in case ";
            cout << "of local maxima)" << endl;
            cout << "Feature set {";
            for (int n = 0; n < current_set_of_features.size(); ++n)
            {
                cout << current_set_of_features.at(n) << ",";
            }
            cout << current_set_of_features.at(current_set_of_features.size() - 1);
            cout << "} was best, accuracy is ";
            cout << best_acc_so_far * 100 << "%" << endl;
            cout << endl;
        }
    }
    
    cout << "Finished search! The best feature subset is {";
    for (int n = 0; n < best_feats.size() - 1; ++n)
    {
        cout << best_feats.at(n) << ",";
    }
    cout << best_feats.at(best_feats.size() - 1);
    cout << "}, which has an accuracy of " << acc_total * 100 << "%" << endl;
    
    return;
}

double loo_nn(vector< vector<double> > data, vector<int> curr_set, int index, 
    bool fwd)
{
    double num_correct = 0;
    
    //for every example
    for (int i = 0; i < data.size(); ++i)
    {
        vector<double> test = data.at(i);
        double min_dist = 999;
        vector<double> min_n;
        
        //find distance from every other example
        for (int j = 0; j < data.size(); ++j)
        {
            if (j != i)
            {
                double dist = e_distance(test, data.at(j), curr_set, index, fwd);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_n = data.at(j);
                }
            }
        }
        
        //sum number of correct guesses
        if ((int)min_n.at(0) == (int)test.at(0))
        {
            ++num_correct;
        }
    }
    
    double tot_acc = num_correct / (double)data.size();
    
    return tot_acc;
}

double e_distance(vector<double> x, vector<double> y, vector<int> curr_set, int index,
    bool fwd)
{
    double dist = 0;
    for (int i = 0; i < curr_set.size(); ++i)
    {
        dist += pow(x.at(curr_set.at(i)) - y.at(curr_set.at(i)), 2);
    }
    if (fwd)
    {
        dist += pow((x.at(index) - y.at(index)),2);
    }
    
    dist = sqrt(dist);
    return dist;
}

int main()
{
    ifstream fin;
    double input = 0;
    vector< vector<double> > data;
    cout << "Welcome to Bootie Wooster's Feature Selection Algorithm." << endl;
    cout << "Type in the name of the file to test: ";
    
    string test_file;
    cin >> test_file;
    cout << endl;
    if (test_file == "CS170_SMALLtestdata__59.txt")
    {
        fin.open("CS170_SMALLtestdata__59.txt");
    }
    else if (test_file == "CS170_LARGEtestdata__107.txt")
    {
        fin.open("CS170_LARGEtestdata__107.txt");
    }
    else
    {
        cout << "File not recognized.";
        return 0;
    }
    if (!fin.is_open())
    {
        cout << "Error opening." << endl;
        return 0;
    }
    else
    {
        for (int i = 0; i < 200; ++i)
        {
            vector<double> temp;
            
            if (test_file == "CS170_LARGEtestdata__107.txt")
            {
            	for (int j = 0; j < 101; ++j)
            	{
                	fin >> input;
                	temp.push_back(input);
            	}
            }
            else if (test_file == "CS170_SMALLtestdata__59.txt")
            {
            	for (int j = 0; j < 11; ++j)
            	{
                	fin >> input;
                	temp.push_back(input);
            	}
            }
            
            data.push_back(temp);
        }
        fin.close();
    }
    
    cout << "Type the number of the algorithm you want to run." << endl;
    cout << "1. Forward Selection" << endl << "2. Backward Elimination"
        << endl << "3. Custom algorithm" << endl;
    int alg;
    cin >> alg;
    
    if (alg == 1)
    {
        forward_selection(data);
    }
    else if (alg == 2)
    {
        backward_elimination(data);
    }
    else
    {
        //Custom Algorithm
        //Since nn is highly susceptible to irrelevant features
        //likely that most are irrelevant
        //test all features individually, and retain top three
        //"quick and dirty" approach that should improve speed
        //at minimal cost to accuracy
        
        custom_alg(data);
    }
    
    
    return 0;
}