package com.example.rogergirgis.wsapplication;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    int clickCount = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        Button btnExperiment2 = (Button) findViewById(R.id.btn_experiment2);
        btnExperiment2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent experimentIntent = new Intent(MainActivity.this, WSActivity.class);
                Bundle configuration = new Bundle();

                configuration.putInt(WSActivity.FOLDER_KEY, clickCount);
                clickCount++;
                experimentIntent.putExtras(configuration);
                MainActivity.this.startActivity(experimentIntent);
            }
        });
    }
}
